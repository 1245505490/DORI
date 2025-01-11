import os
import pickle

from preprocess import save_data
from preprocess.auxiliary import generate_code_levels, generate_code_code_adjacent, getWordEmb
from preprocess.build_dataset import split_patients, build_code_xy
from preprocess.encode import encode_code

if __name__ == '__main__':

    conf = {
        'mimic3': {
            'threshold': 0.01
        },
        'mimic4': {
            'threshold': 0.01,
            'sample_num': 10000
        }
    }
    use_proc = False
    data_path = 'data'
    dataset = 'mimic4'
    seed = 18
    print(f"data: {dataset}, use_proc:{use_proc}")
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if use_proc:
        save_path = os.path.join(dataset_path, 'proc')
    else:
        save_path = os.path.join(dataset_path, 'diag')
    standard_path = os.path.join(save_path, 'standard')
    parsed_path = os.path.join(save_path, 'parsed')
    encoded_path = os.path.join(save_path, 'encoded')


    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)


    if not os.path.exists(raw_path):
        print('please put the CSV files in `data/%s/raw`' % dataset)
        create_dir(raw_path)
        exit()
    create_dir(standard_path)
    create_dir(parsed_path)
    create_dir(encoded_path)

    patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
    admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    p_admission_codes = pickle.load(open(os.path.join(parsed_path, 'p_admission_codes.pkl'), 'rb'))
    patient_disease = pickle.load(open(os.path.join(parsed_path, 'patient_disease.pkl'), 'rb'))
    icd9_maps = pickle.load(open(os.path.join(parsed_path, 'icd9_maps.pkl'), 'rb'))
    patient_dict = pickle.load(open(os.path.join(parsed_path, 'patient_dict.pkl'), 'rb'))
    all_patient = pickle.load(open(os.path.join(parsed_path, 'all_patient.pkl'), 'rb'))
    all_admission_codes = pickle.load(open(os.path.join(parsed_path, 'all_admission_codes.pkl'), 'rb'))

    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
    if use_proc:
        max_procedure_code_num = max([len(codes) for codes in p_admission_codes.values()])
        avg_procedure_code_num = sum([len(codes) for codes in p_admission_codes.values()]) / len(p_admission_codes)
    total_visit_num = sum([len(admissions) for admissions in patient_admission.values()])
    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('total visit num: %d' % total_visit_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    if use_proc:
        print('max code num in an procedure: %d' % max_procedure_code_num)
        print('mean code num in an procedure: %.2f' % avg_procedure_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)

    print('encoding code ...')

    admission_codes_encoded, p_admission_codes_encoded, code_map, pro_code_map = encode_code(patient_admission,
                                                                                             admission_codes,
                                                                                             p_admission_codes)
    code_num = len(code_map) - 1
    pro_code_num = len(pro_code_map) - 1
    print('There are %d codes' % code_num)
    if use_proc:
        print('There are %d pro_codes' % pro_code_num)

    train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map
    )
    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))
    code_levels = generate_code_levels(data_path, pro_code_map)
    code_adj, pro_code_adj = generate_code_code_adjacent(train_pids, code_map, patient_admission,
                                                         admission_codes_encoded,
                                                         p_admission_codes_encoded, code_num,
                                                         pro_code_num, conf[dataset]['threshold'], all_patient,
                                                         all_admission_codes)

    pickle.dump({
        'code_levels': code_levels,
        'code_adj': code_adj,
        'pro_code_adj': pro_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))

    if use_proc:
        max_visit_code_num = max(max_visit_code_num, max_procedure_code_num)
    common_args = [patient_admission, admission_codes_encoded, p_admission_codes_encoded, max_admission_num, code_num,
                   max_visit_code_num]
    print('building train codes features and labels ...')
    train_code_x, train_proc_x, train_code_y, train_visit_lens, train_admtime, train_distime = build_code_xy(train_pids,
                                                                                                             *common_args)
    valid_code_x, valid_proc_x, valid_code_y, valid_visit_lens, valid_admtime, valid_distime = build_code_xy(valid_pids,
                                                                                                             *common_args)
    test_code_x, test_proc_x, test_code_y, test_visit_lens, test_admtime, test_distime = build_code_xy(test_pids,
                                                                                                       *common_args)
    getWordEmb(save_path, icd9_maps, pro_code_map, 64)
    print('saving encoded data ...')
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'admission_codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(pro_code_map, open(os.path.join(encoded_path, 'pro_code_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)

    print('\tsaving training data')
    save_data(train_path, train_code_x, train_proc_x, train_visit_lens, train_code_y, train_admtime, train_distime)
    print('\tsaving valid data')
    save_data(valid_path, valid_code_x, valid_proc_x, valid_visit_lens, valid_code_y, valid_admtime, valid_distime)
    print('\tsaving test data')
    save_data(test_path, test_code_x, test_proc_x, test_visit_lens, test_code_y, test_admtime, test_distime)
