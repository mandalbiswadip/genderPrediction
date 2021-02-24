#include directory paths for the project

HOME = ""
data_path = HOME + '/data'
model_path = HOME + '/model'

max_sentence_length = 20
male_name_path = data_path + "/male_names.txt"
female_name_path = data_path + "/female_names.txt"
loss_file = data_path + '/loss'
seq_model_dir_path = model_path + "/"

alphabets = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']
alphabet_dict = {}

for index,alphabet in  enumerate(alphabets):
    alphabet_dict[alphabet] = index

class_dict ={
    'male':1,
    'female':0
}
