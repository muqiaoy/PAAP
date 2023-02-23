
mkdir -p json_list/dns/tr/
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/synthesized_train/clean > json_list/dns/tr/clean.json
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/synthesized_train/noisy > json_list/dns/tr/noisy.json

mkdir -p json_list/dns/cv/
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/synthesized_valid/clean > json_list/dns/cv/clean.json
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/synthesized_valid/noisy > json_list/dns/cv/noisy.json

mkdir -p json_list/dns/tt/
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/clean > json_list/dns/tt/clean.json
python3 -m datasets.audio /home/muqiaoy/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy > json_list/dns/tt/noisy.json