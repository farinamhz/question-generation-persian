import requests
import json

# from numpy import unicode
import unicodedata


def call_api(url, data, token_key):
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer " + token_key,
        'Cache-Control': "no-cache"
    }
    response = requests.request("POST", url, data=data.encode("utf-8"), headers=headers)
    return response.text


def read_triplets():
    triplets = []
    with open('output.json', 'r') as f:
        for l in f.readlines():
            l = l.replace("\'", "\"")
            # print(l)
            current_line = json.loads(l)
            triplets += current_line['tri']
            # for tup in current_line['tri']:
            #     tup['h']
            #     tup['t']
            #     str(tup['r'][0])
    return triplets


def get_token():
    base_url = "http://api.text-mining.ir/api/"
    url = base_url + "Token/GetToken"
    querystring = {"apikey": "cb0c88a6-b0f7-eb11-80f1-98ded002619b"}
    response = requests.request("GET", url, params=querystring)
    data = json.loads(response.text)
    token_key = data['token']
    return base_url, token_key


def detect(text, base_url, token_key):
    url = base_url + "NamedEntityRecognition/Detect"
    payload = text
    result = json.loads(call_api(url, payload, token_key))
    return result[0]['Tags']['NER']['Item1']


# print(result)
# for phrase in result:
#     print(phrase['Tags']['NER']['Item1'])


if __name__ == '__main__':
    triplets = read_triplets()
    base_url, token_key = get_token()
    questions = []
    answers = []
    unique_list_head = []
    unique_list_tail = []

    for a in triplets:
        print(a['t'])
        # if a['t'] in unique_list_tail and a['h'] in unique_list_head:
        #     continue
        # else:
        unique_list_tail.append(a['t'])
        unique_list_head.append(a['h'])
        # print(a['t'])
        text = a['t']
        text = f'"{text}"'
        entity_type = detect(text, base_url, token_key)
        # print(entity_type)
        s = ''
        if entity_type.find("DAT") != -1:
            s = a['h'] + " در چه زمانی " + str(a['r']) + "؟"
            questions.append(s)
            answers.append(a['t'])
        elif entity_type.find("PER") != -1:
            s = a['h'] + " با چه کسی " + str(a['r']) + "؟"
            questions.append(s)
            answers.append(a['t'])
        elif entity_type.find("LOC") != -1:
            s = a['h'] + " در کجا " + str(a['r']) + "؟"
            questions.append(s)
            answers.append(a['t'])

    for i in range(len(questions)):
        print("سوال: ", questions[i])
        print("جواب: ", answers[i])
