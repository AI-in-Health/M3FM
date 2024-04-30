from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import modules.utils as utils
from modules.caption_model import CaptionModel


class AttModel(CaptionModel):
    def __init__(self, args, tokenizer):
        super(AttModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.max_seq_length = 60

    def _sample(self, clip_features, gpt_tokens,update_opts={}):

        opt = self.args.__dict__
        opt.update(**update_opts)
        sample_method = opt.get('sample_method', 'greedy')


        if sample_method == 'greedy':
            return self._greedy_sample(clip_features, gpt_tokens)
        elif sample_method == 'beam_search':
            return self._beam_search_sample(clip_features, gpt_tokens)
        else:
            raise ValueError("Unknown sample_method: " + sample_method)

    def _greedy_sample(self, clip_features, gpt_tokens, temperature=1.0):
        #input_ids = torch.full((clip_features.size(0), 1), self.tokenizer.bos_token_id).type_as(clip_features).long()
        clip_features = self.clip_project(clip_features).reshape(clip_features.size(0), 1, -1)
        tokens = [None for _ in range(clip_features.size(0))]
        finished = [False for _ in range(clip_features.size(0))]
        max_length = 200
        for _ in range(max_length):
            outputs = self.decoder(inputs_embeds= clip_features)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            next_tokens = torch.argmax(logits, -1).unsqueeze(1)
            next_token_embeds = self.decoder.transformer.wte(next_tokens)
            for j in range(clip_features.size(0)):
                if finished[j]:
                    continue
                if tokens[j] is None:
                    tokens[j] = next_tokens[j]
                else:
                    tokens[j] = torch.cat((tokens[j], next_tokens[j]), dim=0)
                if next_tokens[j].item() == self.tokenizer.eos_token_id:
                    finished[j] = True
            clip_features = torch.cat((clip_features, next_token_embeds), dim=1)
        outputs = []
        for token in tokens:
            try:
                output_list = token.squeeze().cpu().numpy().tolist()
                # Pad or truncate output_list to max_length
                output_list = (output_list + [self.tokenizer.pad_token_id] * max_length)[:max_length]
            except Exception as e:
                print(f"Error during decoding: {type(e).__name__}: {e}")
                output_list = [self.tokenizer.pad_token_id] * max_length
            outputs.append(output_list)

        # Convert list of lists to tensor
        outputs = torch.tensor(outputs, device=clip_features.device)
        return outputs


    def _beam_search_sample(self, clip_features, gpt_tokens, beam_size=5):
        batch_size = clip_features.size(0)
        # Prepare the first input for every beam
        input_ids = torch.full((batch_size*beam_size, 1), self.tokenizer.bos_token_id).type_as(clip_features).long()
        beam_scores = torch.zeros((batch_size, beam_size)).type_as(clip_features)
        done = [False]*batch_size

        for _ in range(self.max_seq_length):
            outputs = self._forward(clip_features.repeat_interleave(beam_size, 0), input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Apply a mask for already finished beams
            next_token_probs[done] = 0
            next_token_probs[:, self.tokenizer.eos_token_id] = -float('Inf')

            # Multiply old scores with new probabilities
            scores = beam_scores.unsqueeze(2) * next_token_probs
            scores = scores.view(batch_size, -1)

            # Get the top beam_size scores and their respective indices
            top_scores, top_indices = scores.topk(beam_size, dim=1)

            # Update beam scores
            beam_scores = top_scores.log()

            # Reshape input_ids
            input_ids = input_ids.view(batch_size, beam_size, -1)

            # Compute next inputs
            next_token_ids = top_indices % self.vocab_size
            beam_indices = top_indices // self.vocab_size
            next_input_ids = torch.cat([input_ids.gather(1, beam_indices.unsqueeze(2).expand(-1, -1, input_ids.size(2))), next_token_ids.unsqueeze(2)], dim=2)

            # Flatten input_ids
            input_ids = next_input_ids.view(batch_size*beam_size, -1)

            # Check which beams are done
            done = (next_token_ids == self.tokenizer.eos_token_id).all(dim=1).tolist()

            if all(done):
                break

        return input_ids.view(batch_size, beam_size, -1)


