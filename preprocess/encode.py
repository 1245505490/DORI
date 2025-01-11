from collections import OrderedDict

from nltk import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def encode_code(patient_admission, admission_codes, p_admission_codes):
    code_map = OrderedDict()
    code_map['PAD'] = 0
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission['adm_id']]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)

    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }

    pro_code_map = OrderedDict(code_map)
    p_admission_codes_encoded = None
    if p_admission_codes is not None:
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = p_admission_codes[admission['adm_id']]
                for code in codes:
                    if code not in pro_code_map:
                        pro_code_map[code] = len(pro_code_map)

        p_admission_codes_encoded = {
            admission_id: list(set(pro_code_map[code] for code in codes))
            for admission_id, codes in p_admission_codes.items()
        }

    return admission_codes_encoded, p_admission_codes_encoded, code_map, pro_code_map


