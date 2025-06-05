#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2024 Fisher. All rights reserved.
#   
#   文件名称：common.py
#   创 建 者：YuLianghua
#   创建日期：2024年12月02日
#   描    述：
#
#================================================================

# QAnything: /qanything_kernel/dependent_server/rerank_server/rerank_backend.py
@get_time
def get_rerank(self, query: str, passages: List[str]):
    tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

    tot_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
        futures = []
        for k in range(0, len(tot_batches), self.batch_size):
            batch = self._tokenizer.pad(
                tot_batches[k:k + self.batch_size],
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors=self.return_tensors
            )
            future = executor.submit(self.inference, batch)
            futures.append(future)
        # debug_logger.info(f'rerank number: {len(futures)}')
        for future in futures:
            scores = future.result()
            tot_scores.extend(scores)

    merge_tot_scores = [0 for _ in range(len(passages))]
    for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
        merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
    # print("merge_tot_scores:", merge_tot_scores, flush=True)
    return merge_tot_scores
