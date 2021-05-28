python3 -m classify --num_iter 999 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/celebA_fraunhofer/img_align_celeba_small \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/celebA_fraunhofer/dataset_celebA \
--experiment_num 1 --save_dataset --saved_file_name celebA_1.pkl

python3 -m classify --num_iter 999 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/celebA_fraunhofer/img_align_celeba_small \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/celebA_fraunhofer/dataset_celebA \
--crop --experiment_num 2 --save_dataset --saved_file_name celebA_2.pkl


python3 -m classify --num_iter 1530 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faceforensics_created/real/c23/jpg-nonresized \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faceforensics_created/deepfakes/c23/jpg-nonresized \
--crop --experiment_num 3 --save_dataset --saved_file_name faceForensics.pkl


python3 -m classify  --num_iter 1000 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/real/combined \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/fake/combined \
--experiment_num 4 --save_dataset --saved_file_name facesHQ_1.pkl


python3 -m classify --num_iter 1000 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/real/combined \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/fake/combined \
--crop --experiment_num 5 --save_dataset --saved_file_name facesHQ_2.pkl

python3 -m classify --num_iter 2000 --reals_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/real/combined \
--fakes_path /home/anaradutoiu/Documents/Sem_2/EAI/datasets/faces-hq/fake/combined \
--experiment_num 6 --save_dataset --saved_file_name facesHQ_3.pkl

python3 -m classify --nocompute_data --saved_data celebA_2.pkl \
--nosplit_dataset --test_file faceForensics.pkl \
--experiment_num 7

python3 -m classify --nocompute_data --saved_data faceForensics.pkl \
--nosplit_dataset --test_file celebA_2.pkl \
--experiment_num 8

python3 -m classify --nocompute_data --saved_data facesHQ_3.pkl \
--nosplit_dataset --test_file celebA_2.pkl \
--experiment_num 9

python3 -m classify --nocompute_data --saved_data celebA_2.pkl \
--nosplit_dataset --test_file facesHQ_3.pkl \
--experiment_num 10

python3 -m classify --nocompute_data --saved_data faceForensics.pkl \
--nosplit_dataset --test_file facesHQ_3.pkl \
--experiment_num 11

python3 -m classify --nocompute_data --saved_data facesHQ_3.pkl \
--nosplit_dataset --test_file faceForensics.pkl \
--experiment_num 12