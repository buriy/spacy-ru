def pluck(dict_list, attr):
    r = []
    for e in dict_list:
        r.append(e[attr])
    return r

def pluck_list(dict_list, *attrs):
    r = []
    for e in dict_list:
        r.append([e[a] for a in attrs])
    return r

def pluck_dict(dict_list, *attrs):
    r = []
    for e in dict_list:
        r.append({a: e[a] for a in attrs})
    return r