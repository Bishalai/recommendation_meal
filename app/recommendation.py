import os
import re
import datetime 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#--------some essential variables
DATAPATH = "datasets/custom"
##users id which is passed 
##these dats are handled by the othr developers
USER_ID = 1
TIME = "breakfast"
RATING = 3
TODAY = datetime.date.today().strftime('%d/%m/%Y')

##-------functions required are stored here
##oads the csv file to the given variable as a pandas object
def load_req_data(filename, data_path=DATAPATH):
    csv_path=os.path.join(data_path, filename)
    return pd.read_csv(csv_path, encoding='cp1252')

def load_data_to_csv(filename, dataframe, data_path=DATAPATH):
    csv_path=os.path.join(data_path, filename)
    dataframe.to_csv(csv_path, mode = 'a',encoding='cp1252', index = False, header=False)
    return


##combines the values of the given columns
def combine_features(row):
    return row['name'] + " " + row['description'] + " " + row['ingredients'] + " " + row['diet']

##gets the name of the food from its index
def get_index_from_name(name):
    return foods[foods.name == name].index.values[0]

##gets the name of the food from its index
def get_name_from_index(index):
    return foods[foods.index==index]['name'].values[0]

##displays the food with its nutritional value in the index inputted(absolute index of the data)
def display_food(i):
    nutrition = foods[foods.index == i]['nutrition'].values[0]
    nut = list(map(float,nutrition.split(',')))
    print(f"{get_name_from_index(i)}: Energy = {nut[0]} Calories, \
Carbohydrate = {nut[1]} gm, Fats = {nut[2]} gm, Protein = {nut[3]} gm ")

#function to compare the value of the ith indexed row's column to the given string
def compare_with_foodvalue(i,column,string ):
    return foods[foods.index == i][column].values[0]==string

def user_datframe(user_id, date, time, food, rating):
    return pd.DataFrame(
    {
        'user_id':[user_id],
        'date':[date],
        'time':[time],
        'food':[rec_food],
        'rating':[rating]
    }
)

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

print('------------------------------\n reaminder after removing based on time')

for item in sorted_similar_foods:
    display_food(item[0])
print("----------------------------\n filtering out last five recommended:")

##filter out last five recommended
past_data = load_req_data("user_data.csv")
past_data['date']=pd.to_datetime(past_data['date'])
past_data.sort_values(['date','user_id'], inplace=True)
past_data = past_data.loc[past_data['user_id'] == USER_ID]
past_data = past_data.tail(5)
print(past_data)
for item in sorted_similar_foods[:]:
    if get_name_from_index(item[0]) in past_data['food'].values:
        print(get_name_from_index(item[0]),"is removed")
        sorted_similar_foods.remove(item)
#get name from index i [0] == past_data['food'].values



print("----------------------------\nremaining foods:")
for item in sorted_similar_foods:
    display_food(item[0])

print("---------------------------\n Final top recommendation")
##show the final result
##nutrition calculation part left
    ###checking or doing calculation of nutrition is left
display_food(sorted_similar_foods[0][0])
rec_food=get_name_from_index(sorted_similar_foods[0][0])

##if the top rec food is a staple food then recommend the highest similar curry
if compare_with_foodvalue(sorted_similar_foods[0][0], 'type', 'staple'):
    for item in sorted_similar_foods:
        if compare_with_foodvalue(item[0], 'type', 'curry'):
            display_food(item[0])
            rec_food = rec_food + ',' + get_name_from_index(item[0])
            break
elif compare_with_foodvalue(sorted_similar_foods[0][0], 'type', 'curry'):
    ####same wise if top recommended is a curry then recommend a companion staple food
    for item in sorted_similar_foods:
        if compare_with_foodvalue(item[0], 'type', 'staple'):
            display_food(item[0])
            rec_food = rec_food + ',' + get_name_from_index(item[0])
            break

print("today's date",TODAY)


##user data: user_id, date, time, food, ratings
updated = user_datframe(USER_ID, TODAY, TIME, rec_food, RATING)

#!!!!!! use this function below if you want to update to csv the current user data
#load_data_to_csv('user_data.csv', updated)



####this is a basic version so far a lot more things need to added and organised which will be done tomorrow
'''
list of things to be added
-nutrition calculation(tougher than I thought)!!!!!
-dynamic upgarding of tags(add words to tag as time passes by)!!!!(someone do it!)
-improved database!
-record history of users in their personal database!!! (DONE!)
-using history remove the recently recommended dishes(must improve catalogue of dishes for this)!!!!(DONE)
-think of a scaling alternative as this doesnt scale well for big datas!!<><>(later!)
-encryption of user's sensitive info!!!(someone do it!)
-make a diseases database that contains ingredients not to use!!!!
-implement these code in their functions!!<><><>
-implement in class if possible, otherwise not necessary!<><><>
'''

##[energy(Cal), carbohydrate(gm),fats(gm),protein(gm)]
# nutrition = foods[foods.index == 2]['nutrition'].values[0]
# nut = list(map(float,nutrition.split(',')))
# print(type(nut))