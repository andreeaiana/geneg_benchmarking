# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/hwwang55/DKN/blob/master/data/kg/prepare_data_for_transx.py

# import libraries
from pathlib import Path

# import custom code
from src.config import FILENAME_KG_FINAL_TXT, FILENAME_TRIPLE2ID, FILENAME_RELATION2ID, FILENAME_ENTITY2ID
from src.util.logger import setup_logging


def prepare_data(kg_in: Path, triple_out: Path, relation_out: Path, entity_out: Path) -> None:
    relation2index = {}
    entity2index = {}
    relation_list = []
    entity_list = []

    reader_kg = open(kg_in, encoding='utf-8')
    writer_triple = open(triple_out, 'w', encoding='utf-8')
    writer_relation = open(relation_out, 'w', encoding='utf-8')
    writer_entity = open(entity_out, 'w', encoding='utf-8')

    entity_cnt = 0
    relation_cnt = 0
    triple_cnt = 0

    logger.info('reading knowledge graph ...')
    kg = reader_kg.read().strip().split('\n')

    logger.info('writing triples to train2id.txt ...')
    writer_triple.write('%d\n' % len(kg))
    for line in kg:
        array = line.split('\t')
        head = array[0]
        relation = array[1]
        tail = array[2]
        if head in entity2index:
            head_index = entity2index[head]
        else:
            head_index = entity_cnt
            entity2index[head] = entity_cnt
            entity_list.append(head)
            entity_cnt += 1
        if tail in entity2index:
            tail_index = entity2index[tail]
        else:
            tail_index = entity_cnt
            entity2index[tail] = entity_cnt
            entity_list.append(tail)
            entity_cnt += 1
        if relation in relation2index:
            relation_index = relation2index[relation]
        else:
            relation_index = relation_cnt
            relation2index[relation] = relation_cnt
            relation_list.append(relation)
            relation_cnt += 1
        writer_triple.write(
            '%d\t%d\t%d\n' % (head_index, tail_index, relation_index))
        triple_cnt += 1
    logger.info('triple size: %d' % triple_cnt)

    logger.info('writing entities to entity2id.txt ...')
    writer_entity.write('%d\n' % entity_cnt)
    for i, entity in enumerate(entity_list):
        writer_entity.write('%s\t%d\n' % (entity, i))
    logger.info('entity size: %d' % entity_cnt)

    logger.info('writing relations to relation2id.txt ...')
    writer_relation.write('%d\n' % relation_cnt)
    for i, relation in enumerate(relation_list):
        writer_relation.write('%s\t%d\n' % (relation, i))
    logger.info('relation size: %d' % relation_cnt)

    reader_kg.close()


if __name__ == '__main__':
    logger = logger = setup_logging(name=__file__, log_level='info')
    prepare_data(kg_in=FILENAME_KG_FINAL_TXT, triple_out=FILENAME_TRIPLE2ID, relation_out=FILENAME_RELATION2ID, entity_out=FILENAME_ENTITY2ID)
