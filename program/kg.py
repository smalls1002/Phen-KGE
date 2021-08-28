COV_disease_list = [
'Phenotype::0W',
'Phenotype::0WGHFT',
'Phenotype::2W',
'Phenotype::4W',
'Phenotype::8W',
'Phenotype::8WGHFT',
'Phenotype::DTF1',
'Phenotype::DTF2',
'Phenotype::DTF3',
'Phenotype::DTFlocSweden2008',
'Phenotype::DTFlocSweden2009',
'Phenotype::DTFmainEffect2008',
'Phenotype::DTFmainEffect2009',
'Phenotype::DTFplantingLoc2008',
'Phenotype::DTFplantingSummer2008',
'Phenotype::DTFplantingSummer2009',
'Phenotype::DTFplantingSummerLocSweden2009',
'Phenotype::DTFspain2008-1',
'Phenotype::DTFspain2008-2',
'Phenotype::DTFspain2009-1',
'Phenotype::DTFspain2009-2',
'Phenotype::DTFsweden2008-1',
'Phenotype::DTFsweden2008-2',
'Phenotype::DTFsweden2009-1',
'Phenotype::DTFsweden2009-2',
'Phenotype::FT10-1',
'Phenotype::FT10',
'Phenotype::FT16-1',
'Phenotype::FT16',
'Phenotype::FT22',
'Phenotype::FTField',
'Phenotype::FTGH',
'Phenotype::LD',
'Phenotype::LDV',
'Phenotype::SD',
'Phenotype::SDV',
'Phenotype::FLC',
'Phenotype::LN10',
'Phenotype::LN16',
'Phenotype::LN22',
'Phenotype::8WGHLN',
'Phenotype::0WGHLN',
'Phenotype::0WGHFT',
'Phenotype::Diameterfield',
'Phenotype::FRI'
]
import os
filepath = "./input/"
fileallname = os.listdir(filepath)
for name in fileallname:

    #pr = name+"############"+"\t"
    #print(pr)

    import csv
    # Load entity file
    drug_list = []
    pathf = filepath +name
    
    with open(pathf, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    len(drug_list)
    treatment = ['AraGWAS::PrG::Phenotype:Gene','KnetMiner::PrG::Phenotype:Gene']
    import pandas as pd
    import numpy as np
    import sys
    #sys.path.insert(1, './utils')
    from utils import download_and_extract
    download_and_extract()

    entity_idmap_file = './entities.tsv'
    relation_idmap_file = './relations.tsv'

    # Get drugname/disease name to entity ID mappings
    entity_map = {}
    entity_id_map = {}
    relation_map = {}
    with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id','name'])
        for row_val in reader:
            entity_map[row_val['name']] = int(row_val['id'])
            entity_id_map[int(row_val['id'])] = row_val['name']

    with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id','name'])
        for row_val in reader:
            relation_map[row_val['name']] = int(row_val['id'])

    # handle the ID mapping
    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_map[drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_map[disease])

    treatment_rid = [relation_map[treat]  for treat in treatment]

    # Load embeddings
    import torch as th
    entity_emb = np.load('./DRKG_TransE_l2_entity.npy')
    rel_emb = np.load('./DRKG_TransE_l2_relation.npy')

    drug_ids = th.tensor(drug_ids).long()
    disease_ids = th.tensor(disease_ids).long()
    treatment_rid = th.tensor(treatment_rid)

    drug_emb = th.tensor(entity_emb[drug_ids])
    treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]

    import torch.nn.functional as fn

    gamma=12.0
    def transE_l2(head, rel, tail):
        score = head + rel - tail
        #print(head,rel,tail,sep="-",end="/n")
        return gamma - th.norm(score, p=2, dim=-1)

    scores_per_disease = []
    dids = []
    for rid in range(len(treatment_embs)):
        treatment_emb=treatment_embs[rid]
        for disease_id in disease_ids:
            disease_emb = entity_emb[disease_id]
            score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
            scores_per_disease.append(score)
            dids.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].numpy()
    dids = dids[idx].numpy()

    _, unique_indices = np.unique(dids, return_index=True)

    #topk=100
    file_path = pathf
    f = open(file_path , "r")
    topk = len(f.readlines())
    f.close()

    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    phename = name.replace(".tsv", "")
    newpath = "./output/"+phename+".txt"
    #print(newpath)
    wf = open(newpath,"w")

    for i in range(topk):
        
        drug = int(proposed_dids[i])
        score = proposed_scores[i]        
        content =  str(entity_id_map[drug])+"\t"+str(score)+"\n"
        wf.write(content)
    wf.close()
