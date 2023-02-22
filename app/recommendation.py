import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#--------some essential variables
DATAPATH = "datasets/custom"
##users id which is passed 
USER_ID = 1
##-------functions required are stored here
def load_req_data(filename, data_path=DATAPATH):
    csv_path=os.path.join(data_path, filename)
    return pd.read_csv(csv_path, encoding='cp1252')

def combine_features(row):
    return row['name'] + " " + row['description'] + " " + row['ingredients'] + " " + row['diet']

def get_index_from_name(name):
    return foods[foods.name == name].index.values[0]

def get_name_from_index(index):
    return foods[foods.index==index]['name'].values[0]


####--- read from csv files
#foods = pd.read_csv("datasets/custom/food.csv", encoding='cp1252')
#users = pd.read_csv("datasets/custom/user_info.csv", encoding="cp1252")
foods = load_req_data("food.csv")
users = load_req_data("user_info.csv")

print("User is: ", users.loc[users['user_id']==USER_ID]['name'])

##----select our tags and features

food_features = ['name','description','ingredients','diet']
user_features = ['tags'] ##the tags must have as many words as possible to describe 
                            #user acurately and must be upated regularly

##----combine features into a new dataframe
features = foods.apply(combine_features, axis = 1)
user_tag = users['tags'].loc[users['user_id']==USER_ID]

##add user tags to the end of list of features
newfeatures = pd.concat([features, user_tag], ignore_index=True)

###-----count matrix
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(newfeatures)

###----compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)

###similar foods in descending order
##take the user's similar data
similar_foods = list(enumerate(cosine_sim[-1]))

##remove the last value since it is 1 i.e. value of user which wont be found in food.csv
similar_foods.remove(similar_foods[-1])
##sort the list in descending order i.e. from most simlar to least similar

sorted_similar_foods = sorted(similar_foods,key=lambda x:x[1],reverse = True)

##show top 5 most recommended
for i in range(0,6):
   print(get_name_from_index(sorted_similar_foods[i][0]))

print("now checking veg")

##-- now filter out the ones that are veg or non veg, allergies and disease
if users[users['user_id']==USER_ID]['diet'].values[0] == "vegetarian":
    for food in sorted_similar_foods[:]:
        if foods[foods.index == food[0]]['diet'].values[0]=="non-vegetarian":
            sorted_similar_foods.remove(food)



# ##check for disease or allergy in ingredients
for foodi in sorted_similar_foods[:]:
    ##make a list of string of the ingredients of food
    ingredients = (foods[foods.index == foodi[0]]['ingredients'].values[0])
    ingred = list(map(str,ingredients.split(',')))
    print(ingred)
    ##check if the list of ingredients in food contains the ingredients user must avoid
    for item in ['onion','carrot']: ##here we use the list of ingredients user should not eat
        if(item in ingred):
            print("removed")
            sorted_similar_foods.remove(foodi)
            break

##sow the final result
##nutrition calculation part left
for i in range(0,6):
    ###checking or doing calculation of nutrition is left
    nutrition = foods[foods.index == sorted_similar_foods[i][0]]['nutrition'].values[0]
    nut = list(map(float,nutrition.split(',')))
    print(f"{get_name_from_index(sorted_similar_foods[i][0])}: Energy={nut[0]} Calories, \
Carbohydrate = {nut[1]} gm, Fats = {nut[2]} gm, Protein = {nut[3]} gm ")
    
####this is a basic version so far a lot more things need to added and organised which will be done tomorrow


##[energy(Cal), carbohydrate(gm),fats(gm),protein(gm)]
# nutrition = foods[foods.index == 2]['nutrition'].values[0]
# nut = list(map(float,nutrition.split(',')))
# print(type(nut))
