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
TIME = "breakfast"
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

def display_food(i, sorted_food_list):
    nutrition = foods[foods.index == sorted_food_list[i][0]]['nutrition'].values[0]
    nut = list(map(float,nutrition.split(',')))
    print(f"{get_name_from_index(sorted_food_list[i][0])}: Energy={nut[0]} Calories, \
Carbohydrate = {nut[1]} gm, Fats = {nut[2]} gm, Protein = {nut[3]} gm ")



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
print("first batch of recommended without any constraints")
for i in range(0,6):
   print(get_name_from_index(sorted_similar_foods[i][0]))
print("----------------------------------------------")
print("now removing the foods based on ingredients")

##-- now filter out the ones that are veg or non veg, allergies and disease
if users[users['user_id']==USER_ID]['diet'].values[0] == "vegetarian":
    for food in sorted_similar_foods[:]:
        if foods[foods.index == food[0]]['diet'].values[0]=="non-vegetarian":
            print(get_name_from_index(food[0]), "is removed")
            sorted_similar_foods.remove(food)



# ##check for disease or allergy in ingredients
for food in sorted_similar_foods[:]:
    ##make a list of string of the ingredients of food
    ingredients = (foods[foods.index == food[0]]['ingredients'].values[0])
    ingred = list(map(str,ingredients.split(',')))
    print(ingred)
    ##check if the list of ingredients in food contains the ingredients user must avoid
    for item in ['carrot','chicken']: ##here we use the list of ingredients user should not eat
        if(item in ingred):
            print(get_name_from_index(food[0])," is removed")
            sorted_similar_foods.remove(food)
            break

print("-----------------------------\nremoving based on time")
##check for the food time i.e. foods associated with lunch is only taken during lunch
for food in sorted_similar_foods[:]:
    time_food = (foods[foods.index == food[0]]['time'].values[0])
    time_food_list = list(map(str,time_food.split(',')))
    if TIME not in time_food_list:
        print(get_name_from_index(food[0])," is removed")
        sorted_similar_foods.remove(food)


print("----------------------------\nremaining foods:")
for i in range(len(sorted_similar_foods)):
    display_food(i, sorted_similar_foods)


print("---------------------------\n Final top recommendation")
##show the final result
##nutrition calculation part left
    ###checking or doing calculation of nutrition is left
display_food(0, sorted_similar_foods)


##if the top rec food is a staple food then recommend the highest similar curry
if foods[foods.index == sorted_similar_foods[0][0]]['type'].values[0]=="staple":
    for i in range(len(sorted_similar_foods)):
        if foods[foods.index == sorted_similar_foods[i][0]]['type'].values[0]=="curry":
            display_food(i, sorted_similar_foods)
            break
elif foods[foods.index == sorted_similar_foods[0][0]]['type'].values[0]=="curry":
    ####same wise if top recommended is a curry then recommend a companion staple food
    for i in range(len(sorted_similar_foods)):
        if foods[foods.index == sorted_similar_foods[i][0]]['type'].values[0]=="staple":
            display_food(i, sorted_similar_foods)
            break



####this is a basic version so far a lot more things need to added and organised which will be done tomorrow
'''
list of things to be added
-nutrition!!!!!
-dyanmic upgarding of tags(add words to tag as time passes by)!!!!
-improved database!
-record history of users in their personal database!!!
-using history remove the recently recommended dishes(must improve catalogue of dishes for this)!!!!
-think of a scaling alternative as this doesnt scale well for big datas!!
-encryption of user's sensitive info!!!
-make a diseases database that contains ingredients not to use!!!!
-implement these code in their functions!!
-implement in class if possible, otherwise not necessary!
'''

##[energy(Cal), carbohydrate(gm),fats(gm),protein(gm)]
# nutrition = foods[foods.index == 2]['nutrition'].values[0]
# nut = list(map(float,nutrition.split(',')))
# print(type(nut))
