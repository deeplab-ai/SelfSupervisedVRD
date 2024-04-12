# -*- coding: utf-8 -*-
"""A custom loader of annotations using multiple filters."""
import copy
import sys
from collections import Counter
from copy import deepcopy
import json
import os
import random
from tqdm import tqdm

import numpy as np


class AnnotationLoader:
    """A class to load and filter annotations."""

    def __init__(self, config):
        """Initialize loader."""
        self.config = config
        self._set_from_config(config)
        self.reset()
        self.logger = config.logger

    def _set_from_config(self, config):
        """Load config variables."""
        self._bg_perc = config.bg_perc
        self._lb_perc = config.lb_perc
        self._lb_imgs = config.lb_imgs
        self._train_on = config.train_on
        self.random_few_shot = config.random_few_shot
        self.num_rel_classes = config.num_rel_classes
        self._k_shot = config.k_shot
        self.random_seed = config.random_seed
        self.model_name = config.net_name
        self._dataset = config.dataset
        self._classes_to_keep = config.classes_to_keep
        self._filter_duplicate_rels = config.filter_duplicate_rels
        self._filter_multiple_preds = config.filter_multiple_preds
        self._json_path = config.paths['json_path']
        self._mode = config.task
        self._orig_images_path = config.orig_img_path
        self._pairs_limit = config.relations_per_img_limit
        self._train_with_negatives = config.use_negative_samples
        self._test_with_negatives = config.test_on_negatives
        # self.prerequisites_path = config.prerequisites_path
        self.prerequisites_path = 'prerequisites/'

    def reset(self, mode=None):
        """Reset loader with new mode."""
        self._annos = []
        if mode is not None:
            self._mode = mode

    def get_annos(self):
        """Return full filtered annotations."""
        if not self._annos:
            self._annos = self._load_annotations()
            if not self._mode.startswith('obj'):
                self._annos = self._filter_annotations(self._annos)
            if self._k_shot:
                self._annos = self._few_shot_filter(self._annos, self._k_shot)
            if self._lb_imgs:
                self._annos = self.create_splits(self._annos)
            if self._train_on:
                self._annos = self.train_num_images(self._annos)
            if self.random_few_shot:
                self._annos = self.random_few_shot_filter(self._annos)
        return self._annos

    def random_few_shot_filter(self, annos):
        self.logger.debug(f"Performing {self.random_few_shot}-shot learning with random sampling..")
        if not self.random_seed:
            raise AttributeError("Please configure a seed for few-shot sampling")
        else:
            random.Random(self.random_seed).shuffle(annos)
        random_shot_annos = []
        pred_count = {i: 0 for i in range(self.num_rel_classes)}

        static_keys = ['filename', 'split_id', 'height', 'width', 'dataset', 'objects']
        # rel_keys = ['ids', 'merged_ids', 'names', 'subj_ids', 'obj_ids', 'neg_ids']
        # obj_keys = ['boxes', 'ids', 'names', 'scores']
        rel_keys = ['ids', 'merged_ids', 'names', 'subj_ids', 'obj_ids']
        obj_keys = ['boxes', 'ids', 'names']

        for anno in annos:
            if anno['split_id'] == 0:
                new_anno = {key: anno[key] for key in static_keys}
                new_anno['relations'] = {key: np.array([], dtype=int) for key in rel_keys}
                new_anno['relations']['neg_ids'] = anno['relations']['neg_ids']
                for i, ind in enumerate(anno['relations']['ids']):
                    if pred_count[ind] < self.random_few_shot:
                        for rel_key in rel_keys:
                            new_anno['relations'][rel_key] = np.concatenate((new_anno['relations'][rel_key],
                                                                             np.array(
                                                                                 [anno['relations'][rel_key][i]])))
                        pred_count[ind] += 1
                    else:
                        continue
                if len(new_anno['relations']['ids']) > 0:
                    random_shot_annos.append(new_anno)
            else:
                random_shot_annos.append(anno)
        num_labeled_data = 0
        for annos in random_shot_annos:
            if annos['split_id'] == 0:
                for ind in annos['relations']['ids']:
                    if (ind != 70 and self._dataset == 'VRD') or (ind != 50 and self._dataset == 'VG200'):
                        num_labeled_data += 1
        self.logger.info(f"Num. of labeled data: {num_labeled_data}")

        return random_shot_annos

    def train_num_images(self, annos):
        if self.random_seed:
            random.Random(self.random_seed).shuffle(annos)
        else:
            raise AttributeError("Please specify a seed!!")
        i = 0
        reduced_annos = []
        for idx, anno in enumerate(annos):
            if anno['split_id'] == 0:
                i += 1
                if i <= self._train_on:
                    reduced_annos.append(anno)
                else:
                    continue
            else:
                reduced_annos.append(anno)
        num_training_data = 0
        for anno in reduced_annos:
            if anno['split_id'] == 0:
                num_training_data += 1
        self.logger.debug(f"Number of training data: {num_training_data}")
        return reduced_annos

    def create_splits(self, annos):
        annos_splited = 0
        if self.random_seed:
            random.Random(self.random_seed).shuffle(annos)
        else:
            raise AttributeError("Please specify a seed!!")
        i = 0
        for idx, anno in enumerate(annos):
            if anno['split_id'] == 0:
                i += 1
                if i <= self._lb_imgs:
                    # print(idx)
                    anno['labeled'] = 1
                else:
                    # print(idx)
                    anno['labeled'] = 0
            else:
                anno['labeled'] = 2
        return annos

    def _few_shot_filter(self, annos, n):
        self.logger.info(f"Creating {n}-shot dataset...")
        if self._mode == 'preddet':
            # json_v = '2' if self._dataset == 'VG200' else '3'
            # self.logger.debug(f"Loading {self._dataset}_few_shot_dict_v{json_v}.json file...")
            self.logger.debug(f"Loading {self._dataset}_few_shot_dict.json file...")
            with open(self.prerequisites_path + self._dataset + f"_few_shot_dict.json", 'r') as f:
                few_shot_dict = json.load(f)
        elif self._mode == 'predcls':
            self.logger.debug(f"Loading {self._dataset}_few_shot_dict_predcls.json file...")
            with open(self._json_path + self._dataset + f"_few_shot_dict_predcls.json", 'r') as f:
                few_shot_dict = json.load(f)
        else:
            raise AttributeError(f"Task {self._mode} isn't implemented yet")

        to_keep = {}
        for key in few_shot_dict:
            for i in range(n):
                if len(few_shot_dict[key]) - 1 >= i:
                    idx = few_shot_dict[key][i][0]
                    rel_idx = few_shot_dict[key][i][1]

                    if idx not in to_keep:
                        to_keep[idx] = set()
                    to_keep[idx].add(rel_idx)
        # print("Filtering annos...")
        few_shot_annos = []
        for i in tqdm(range(len(annos))):
            if annos[i]['split_id'] != 0:
                few_shot_annos.append(annos[i])
            else:
                if i in to_keep:
                    new_anno = {}
                    for k in ['filename', 'split_id', 'height', 'width', 'dataset', 'objects']:
                        new_anno[k] = annos[i][k]

                    new_anno['relations'] = {}
                    for k in annos[i]['relations']:
                        if annos[i]['relations'][k] is not None and len(annos[i]['relations'][k]) > 0:
                            new_anno['relations'][k] = annos[i]['relations'][k][list(to_keep[i])]
                        else:
                            new_anno['relations'][k] = annos[i]['relations'][k]
                    few_shot_annos.append(new_anno)
        num_labeled_data = 0
        preds_count = {i: 0 for i in range(71)}
        # print(len(few_shot_annos))
        for annos in few_shot_annos:
            if annos['split_id'] == 0:
                for ind in annos['relations']['ids']:
                    preds_count[ind] += 1
                    if (ind != 70 and self._dataset == 'VRD') or (ind != 50 and self._dataset == 'VG200'):
                        num_labeled_data += 1
        self.logger.info(f"Num. of labeled data: {num_labeled_data}")
        # print("Done!")
        return few_shot_annos

    def _reduce_labaled_data(self, annotations):
        if self._lb_perc:
            preds_count = {i: 0 for i in range(71)}
            red_preds_count = {i: 0 for i in range(71)}
            for annos in annotations:
                # print(annos['relations'])
                for ind in annos['relations']['ids']:
                    if ind != 70:
                        preds_count[ind] += 1
            # print(preds_count)
            num_labeled = 0
            for key in preds_count.keys():
                if key != 70:
                    num_labeled += preds_count[key]
            # print('num of labeled data: ', num_labeled)
            self.logger.debug("Num. of labeled data: %d", num_labeled)
            self.logger.info("Reducing labeled data...")
            lb_perc = self._lb_perc
            red_lb = int(lb_perc * num_labeled)
            # print('reduction of labeled data by: ', red_lb)
            new_num_labaled = (1.0 - lb_perc) * num_labeled
            # print('new num of labeled data: ', int(new_num_labaled))
            self.logger.debug("New num. of labeled data: %d", new_num_labaled)
            sorted_preds = {k: preds_count[k] for k in sorted(preds_count, key=preds_count.get, reverse=True)}
            # print(sorted_preds)

            first_key = list(sorted_preds.keys())[0]
            while red_lb > 0:
                for key in sorted_preds.keys():
                    tmp = sorted_preds[key]
                    if sorted_preds[first_key] > 999 and tmp > 999:
                        if int(tmp / 2) > red_lb:
                            red_preds_count[key] += red_lb
                            sorted_preds[key] -= red_lb
                            red_lb = 0
                        else:
                            red_lb -= int(tmp / 2)
                            red_preds_count[key] += int(tmp / 2)
                            sorted_preds[key] = int(tmp / 2)
                    elif 99 < sorted_preds[first_key] < 999 and tmp > 99:
                        if int(tmp / 2) > red_lb:
                            red_preds_count[key] += red_lb
                            sorted_preds[key] -= red_lb
                            red_lb = 0
                        else:
                            red_lb -= int(tmp / 2)
                            red_preds_count[key] += int(tmp / 2)
                            sorted_preds[key] = int(tmp / 2)
                    elif 9 < sorted_preds[first_key] < 99 and tmp > 9:
                        if int(sorted_preds[key] / 2) > red_lb:
                            red_preds_count[key] += red_lb
                            sorted_preds[key] -= red_lb
                            red_lb = 0
                        else:
                            red_lb -= int(tmp / 2)
                            red_preds_count[key] += int(tmp / 2)
                            sorted_preds[key] = int(tmp / 2)
            red_preds_count = {k: red_preds_count[k] for k in
                               sorted(red_preds_count, key=red_preds_count.get, reverse=True)}
            # print('red_preds_count: ', red_preds_count)
            # print('sorted_preds: ', sorted_preds)
            # tmp1 = tmp2 = 0
            # for key in red_preds_count.keys():
            #     tmp2 += sorted_preds[key]
            #     tmp1 += red_preds_count[key]
            # print(tmp1, tmp2)
            # print(annotations[100])
            for annos in annotations:
                # print(annos['relations'])
                cntr = 0
                for indx, id in enumerate(annos['relations']['ids']):
                    if red_preds_count[id] > 0:
                        # delete the reduced labeled data
                        annos['relations']['ids'] = np.delete(annos['relations']['ids'], indx - cntr)
                        annos['relations']['names'] = np.delete(annos['relations']['names'], indx - cntr)
                        annos['relations']['merged_ids'] = np.delete(annos['relations']['merged_ids'], indx - cntr)
                        annos['relations']['subj_ids'] = np.delete(annos['relations']['subj_ids'], indx - cntr)
                        annos['relations']['obj_ids'] = np.delete(annos['relations']['obj_ids'], indx - cntr)

                        # replace the reduced labeled data with bg
                        # annos['relations']['ids'][indx] = 70
                        # annos['relations']['names'][indx] = '__background__'
                        red_preds_count[id] -= 1
                        cntr += 1
            # remove images with zero relations
            empty_images = []
            for indx, annos in enumerate(annotations):
                if annos['relations']['names'].size == 0:
                    empty_images.append(indx)
            for i, empty_image in enumerate(empty_images):
                del annotations[empty_image - i]
            # print(annotations[100])
            return annotations

    def few_shot(self, annotations):
        self.logger.info(f"{self._k_shot}-shot learning...")
        # print(annotations[0])
        preds_count = {i: 0 for i in range(71)}
        red_preds_count = {i: self._k_shot for i in range(70)}
        red_preds_count[70] = 0
        keep_rel_few_shot = {f"{i}": [] for i in range(len(annotations)) if annotations[i]['split_id'] == 0}
        # for annos in annotations:
        #     # print(annos['relations'])
        #     for ind in annos['relations']['ids']:
        #         if ind != 70:
        #             preds_count[ind] += 1
        # print(preds_count)
        # print(red_preds_count)
        # sorted_rel = {k: preds_count[k] for k in
        #  sorted(preds_count, key=preds_count.get, reverse=True)}
        # print(sorted_rel)
        # num_labeled = 0
        # for key in preds_count.keys():
        #     if key != 70:
        #         num_labeled += preds_count[key]
        # self.logger.debug(f"Shuffling with seed of value {self.random_seed}")
        # random.Random(self.random_seed).shuffle(annotations)
        with open(self._json_path + self._dataset + "_few_shot_dict_v2.json", 'r') as f:
            few_shot_dict_samples = json.load(f)
        # print(few_shot_dict_samples)
        # print(annotations[73]['relations']['ids'][14])  # the class of the predicate (e.g. 0)
        # print(annotations[0])

        for class_idx in few_shot_dict_samples:
            for idx, samples in enumerate(few_shot_dict_samples[class_idx]):
                keep_rel_few_shot[f"{samples[0]}"].append(samples[1])
                if idx + 1 == self._k_shot:
                    break
        # few_shot_annos = []
        # print(annotations[0]['relations'])
        # print(annotations[0])
        # for key in keep_rel_few_shot.keys():
        #     few_shot_annos.append(
        #         {
        #             'filename': annotations[int(key)]['filename'],
        #
        #         }
        #     )
        #     print(keep_rel_few_shot[key])
        #     break
        # sys.exit(2)
        # print(
        #     annotations[keep_rel_few_shot['0']]['relations']['ids'][0]
        # )

        # sys.exit(2)
        # keep the k first relations
        # for idx, annos in enumerate(annotations):
        #     if annos['split_id'] == 0:
        #         cntr = 0
        #         for indx, id in enumerate(annos['relations']['ids']):
        #             if red_preds_count[id] > 0:
        #                 red_preds_count[id] -= 1
        #             else:
        #                 annos['relations']['ids'] = np.delete(annos['relations']['ids'], indx - cntr)
        #                 annos['relations']['names'] = np.delete(annos['relations']['names'], indx - cntr)
        #                 annos['relations']['merged_ids'] = np.delete(annos['relations']['merged_ids'], indx - cntr)
        #                 annos['relations']['subj_ids'] = np.delete(annos['relations']['subj_ids'], indx - cntr)
        #                 annos['relations']['obj_ids'] = np.delete(annos['relations']['obj_ids'], indx - cntr)
        #                 cntr += 1

        # sotiris dict (thanks sotiris)
        few_shot_annos = []
        for idx, annos in enumerate(annotations):
            if annos['split_id'] == 0:
                cntr = 0
                for indx, id in enumerate(annos['relations']['ids']):
                    if indx in keep_rel_few_shot[f"{idx}"]:
                        continue
                    else:
                        annos['relations']['ids'] = np.delete(annos['relations']['ids'], indx - cntr)
                        annos['relations']['names'] = np.delete(annos['relations']['names'], indx - cntr)
                        annos['relations']['merged_ids'] = np.delete(annos['relations']['merged_ids'], indx - cntr)
                        annos['relations']['subj_ids'] = np.delete(annos['relations']['subj_ids'], indx - cntr)
                        annos['relations']['obj_ids'] = np.delete(annos['relations']['obj_ids'], indx - cntr)
                        cntr += 1
                if annos['relations']['names'].size > 0:
                    few_shot_annos.append(annos)
            else:
                few_shot_annos.append(annos)
        # empty_images = []
        # for indx, annos in enumerate(annotations):
        #     if annos['relations']['names'].size == 0:
        #         empty_images.append(indx)
        # for i, empty_image in enumerate(empty_images):
        #     del annotations[empty_image - i]

        num_labeled_data = 0
        preds_count = {i: 0 for i in range(71)}
        # print(len(few_shot_annos))
        for annos in few_shot_annos:
            if annos['split_id'] == 0:
                for ind in annos['relations']['ids']:
                    preds_count[ind] += 1
                    if (ind != 70 and self._dataset == 'VRD') or (ind != 50 and self._dataset == 'VG200'):
                        num_labeled_data += 1
        self.logger.info(f"Num. of labeled data: {num_labeled_data}")
        # print("num of images :", len(annotations))
        # print(few_shot_annos[0])
        # sys.exit(2)
        return few_shot_annos

    def get_class_counts(self, feature='relations'):
        """Return class frequencies for relations or objects."""
        annos = self.get_annos()
        cntr = Counter([_id for anno in annos for _id in anno[feature]['ids']])
        return np.array([cntr[_id] for _id in sorted(list(cntr.keys()))])

    def get_zs_annos(self):
        """Return zero-shot annotations."""
        if not self._annos:
            self._annos = self._load_annotations()
            self._annos = self._filter_annotations(self._annos)
        seen = set(
            (anno['objects']['ids'][sid], rel_id, anno['objects']['ids'][oid])
            for anno in self._annos if anno['split_id'] == 0
            for sid, rel_id, oid in zip(
                anno['relations']['subj_ids'],
                anno['relations']['ids'],
                anno['relations']['obj_ids']
            )
        )
        zs_annos = []
        for anno in self._annos:
            if anno['split_id'] == 2 and anno['relations']['names'].tolist():
                keep = [
                    r for r, (sid, rid, oid) in enumerate(zip(
                        anno['objects']['ids'][anno['relations']['subj_ids']],
                        anno['relations']['ids'],
                        anno['objects']['ids'][anno['relations']['obj_ids']]
                    ))
                    if (sid, rid, oid) not in seen
                ]
                if keep:
                    anno = deepcopy(anno)
                    anno['relations'] = {
                        'ids': anno['relations']['ids'][keep],
                        'merged_ids': anno['relations']['merged_ids'][keep],
                        'names': anno['relations']['names'][keep],
                        'subj_ids': anno['relations']['subj_ids'][keep],
                        'obj_ids': anno['relations']['obj_ids'][keep]
                    }
                    zs_annos.append(anno)
        return zs_annos

    def _filter_annotations(self, annotations):
        """Apply specified filters on annotations."""
        # Enhance with negatives
        annotations = self._merge_negatives(annotations)
        # Ensure there are foreground samples if preddet, no bg in eval
        if self._mode == 'preddet':
            annotations = [
                self._filter_bg(anno) if anno['split_id'] != 0 else anno
                for anno in annotations
                if len(set(anno['relations']['names'].tolist())) > 1
            ]
        # Keep only samples of specific classes
        if self._classes_to_keep is not None:
            for anno in annotations:
                if anno['split_id'] == 2:
                    anno['relations'] = self._filter_nontail(anno['relations'])
        annotations = [
            anno for anno in annotations if anno['relations']['names'].tolist()
        ]
        # Filter duplicate triplets per pair
        if self._filter_duplicate_rels:
            for anno in annotations:
                if anno['split_id'] == 0:
                    anno['relations'] = self._filter_dupls(anno['relations'])
        # Filter multiple triplets per pair
        if self._filter_multiple_preds:
            for anno in annotations:
                if anno['split_id'] == 0:
                    anno['relations'] = self._filter_multi(anno['relations'])
        # Keep at most relations_per_img_limit pairs for memory issues
        for anno in annotations:
            if anno['split_id'] == 0:
                anno['relations'] = self._filter_pairs(anno['relations'])
        # Reduce labeled data
        if self._lb_perc:
            annotations = self._reduce_labaled_data(annotations)
        # if self._k_shot and 'pretrained' not in self.model_name:
        #     annotations = self.few_shot(annotations)
        # sys.exit()

        return annotations

    @staticmethod
    def _filter_bg(anno):
        """Filter background annotations."""
        inds = np.array([
            n for n, name in enumerate(anno['relations']['names'])
            if name != '__background__' or anno['relations']['neg_ids'][n]
        ])
        anno['relations']['names'] = anno['relations']['names'][inds]
        anno['relations']['ids'] = anno['relations']['ids'][inds]
        anno['relations']['merged_ids'] = anno['relations']['merged_ids'][inds]
        anno['relations']['subj_ids'] = anno['relations']['subj_ids'][inds]
        anno['relations']['obj_ids'] = anno['relations']['obj_ids'][inds]
        anno['relations']['neg_ids'] = anno['relations']['neg_ids'][inds]
        return anno

    @staticmethod
    def _filter_dupls(relations):
        """Filter relations appearing more than once."""
        _, unique_inds = np.unique(np.stack(
            (relations['subj_ids'], relations['ids'], relations['obj_ids']),
            axis=1
        ), axis=0, return_index=True)
        return {
            'ids': relations['ids'][unique_inds],
            'merged_ids': relations['merged_ids'][unique_inds],
            'names': relations['names'][unique_inds],
            'subj_ids': relations['subj_ids'][unique_inds],
            'obj_ids': relations['obj_ids'][unique_inds],
            'neg_ids': relations['neg_ids'][unique_inds]
        }

    @staticmethod
    def _filter_multi(relations):
        """Filter multiple annotations for the same object pair."""
        _, unique_inds = np.unique(np.stack(
            (relations['subj_ids'], relations['obj_ids']), axis=1
        ), axis=0, return_index=True)
        return {
            'ids': relations['ids'][unique_inds],
            'merged_ids': relations['merged_ids'][unique_inds],
            'names': relations['names'][unique_inds],
            'subj_ids': relations['subj_ids'][unique_inds],
            'obj_ids': relations['obj_ids'][unique_inds],
            'neg_ids': relations['neg_ids'][unique_inds]
        }

    def _filter_nontail(self, relations):
        """Filter non-tail classes."""
        keep_inds = relations['ids'][:, None] == self._classes_to_keep[None, :]
        keep_inds = keep_inds.any(1)
        return {
            'ids': relations['ids'][keep_inds],
            'merged_ids': relations['merged_ids'][keep_inds],
            'names': relations['names'][keep_inds],
            'subj_ids': relations['subj_ids'][keep_inds],
            'obj_ids': relations['obj_ids'][keep_inds],
            'neg_ids': relations['neg_ids'][keep_inds]
        }

    def _filter_pairs(self, relations):
        """Limit pairs per image for memory issues."""
        relations['names'] = relations['names'][:self._pairs_limit]
        relations['ids'] = relations['ids'][:self._pairs_limit]
        relations['merged_ids'] = relations['merged_ids'][:self._pairs_limit]
        relations['subj_ids'] = relations['subj_ids'][:self._pairs_limit]
        relations['obj_ids'] = relations['obj_ids'][:self._pairs_limit]
        relations['neg_ids'] = relations['neg_ids'][:self._pairs_limit]
        return relations

    def _load_annotations(self):
        """Load annotations from json."""
        _mode = '_sggen_merged' if self._mode == 'sggen' else '_predcls'
        if self._dataset in {'VG80K', 'VrR-VG'}:
            _mode = '_preddet'
        with open(self._json_path + self._dataset + _mode + '.json') as fid:
            annotations = json.load(fid)
        return self._to_list_with_arrays(annotations)

    def _merge_negatives(self, annotations):
        """Merge with negative labels."""
        negatives = {}
        if self._train_with_negatives or self._test_with_negatives:
            neg_json = self._json_path + self._dataset + '_negatives.json'
            with open(neg_json) as fid:
                negatives = json.load(fid)
        for anno in annotations:
            neg_ids = [[] for _ in range(len(anno['relations']['ids']))]
            update_neg_ids = (
                    anno['filename'] in negatives
                    and (
                            (anno['split_id'] < 2 and self._train_with_negatives)
                            or (anno['split_id'] == 2 and self._test_with_negatives)
                    )
            )
            if update_neg_ids:
                neg_ids = negatives[anno['filename']]
            anno['relations']['neg_ids'] = np.copy(neg_ids)
        return annotations

    def _to_list_with_arrays(self, annotations):
        """Transform lists to numpy arrays."""
        orig_img_names = set(os.listdir(self._orig_images_path))
        if self._dataset == 'COCO':  # COCO training mines from VRD/VG images
            orig_img_names = orig_img_names.union(
                set(os.listdir(self._orig_images_path.replace('COCO', 'VRD')))
            )
            orig_img_names = orig_img_names.union(
                set(os.listdir(self._orig_images_path.replace('COCO', 'VG')))
            )
        return [
            {
                'filename': anno['filename'],
                'split_id': anno['split_id'],
                'height': anno['height'],
                'width': anno['width'],
                'dataset': anno['dataset'] if 'dataset' in anno else None,
                'objects': {
                    'boxes': np.array(anno['objects']['boxes']),
                    'ids': np.array(anno['objects']['ids']).astype(int),
                    'names': np.array(anno['objects']['names']),
                    'scores': (
                        np.array(anno['objects']['scores'])
                        if 'scores' in anno['objects']
                           and anno['objects']['scores'] is not None
                        else None)
                },
                'relations': {
                    'ids': np.array(anno['relations']['ids']).astype(int),
                    'merged_ids': np.array(
                        anno['relations']['merged_ids']).astype(int),
                    'names': np.array(anno['relations']['names']),
                    'subj_ids': np.array(anno['relations']['subj_ids']),
                    'obj_ids': np.array(anno['relations']['obj_ids'])
                }
            }
            for anno in annotations
            if anno['filename'] in orig_img_names
               and (any(anno['relations']['names']) or 'obj' in self._mode)
               and any(anno['objects']['names'])
               and not ('dataset' in anno and anno['dataset'] == 'COCO')
        ]
