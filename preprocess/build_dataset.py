from datetime import datetime

import numpy as np


def split_patients(patient_admission, admission_codes, code_map, seed=18):
    np.random.seed(seed)
    common_pids = set()
    print("Split patients...")
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission['adm_id']]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)
    train_num = int(len(patient_admission) * 0.8)
    valid_num = int(len(patient_admission) * 0.1)
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, p_admission_codes_encoded, max_admission_num,
                  code_num, max_visit_code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_visit_code_num), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    pro_x = np.zeros((n, max_admission_num, max_visit_code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    adm_time = np.zeros((n, max_admission_num), dtype=float)
    dis_time = np.zeros_like(adm_time, dtype=float)
    st = datetime(1970, 1, 1)

    def get_time_relative(admission_time, first_time, max_time):
        time_seconds = (admission_time - st).total_seconds()
        return (time_seconds - first_time) / max_time if max_time != 0 else 0

    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        first_admtime = (admissions[0]['adm_time'] - st).total_seconds()
        first_distime = (admissions[0]['adm_time'] - st).total_seconds()
        max_adm_time = (admissions[-2]['dis_time'] - st).total_seconds() - first_admtime
        max_dis_time = (admissions[-2]['dis_time'] - st).total_seconds() - first_distime
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            x[i, k, :len(codes)] = codes

            if p_admission_codes_encoded is not None:
                pro_codes = p_admission_codes_encoded[admission['adm_id']]
                pro_x[i, k, :len(pro_codes)] = pro_codes

            if max_adm_time != 0:
                adm_time[i, k] = get_time_relative(admission['adm_time'], first_admtime, max_adm_time)
            if max_dis_time != 0:
                dis_time[i, k] = get_time_relative(admission['adm_time'], first_distime, max_dis_time)

        codes = np.array(admission_codes_encoded[admissions[-1]['adm_id']]) - 1
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
        if max_adm_time != 0:
            adm_time[i, lens[i] - 1] = 1
        else:
            adm_time[i, :lens[i]] = np.linspace(0, 1, lens[i])
        if max_dis_time != 0:
            dis_time[i, lens[i] - 1] = 1
        else:
            dis_time[i, :lens[i]] = np.linspace(0, 1, lens[i])
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, pro_x, y, lens, adm_time, dis_time
