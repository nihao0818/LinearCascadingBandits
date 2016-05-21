""".py"""
from pyspark import SparkContext
import json

# ~/Documents/spark-1.3.0-bin-hadoop2.4/bin/spark-submit preprocessing_yelp_data.py 

def fetch(line):
	data = json.loads(line)
	return (data['date'],1)
	# return (data['business_id'],data['user_id'])	
	# return (data['business_id'],1)


def convert2csv(pair):
	r = 'Business_id,' + str(pair[0])
	r += ',User_id'
	for i in pair[1]:
		r += ',' + str(i)
	return r

def fetchBusinessDate(line):
	data = json.loads(line)
	return ((data['business_id'],data['date']),1)

	# user not date


def fetchBusinessUserPair(line):
	data = json.loads(line)
	return ((data['business_id'],data['user_id']),1)


def fetchResturant(line):
	data = json.loads(line)
	categories = data['categories']
	categoriesStr = ','.join(categories)
	if "Restaurants" in categories:
		return (data['business_id'],categoriesStr)
	else:
		return ("NOTRESTAURANT",1)

def fetchSpecificResturant(line):
	data = json.loads(line)
	categories = data['categories']
	categoriesStr = ','.join(categories)
	if "Restaurants" in categories:
		if "American (New)" in categories:
			return (data['business_id'],categoriesStr)
		else:
			return ("NOTRESTAURANT",1)
	else:
		return ("NOTRESTAURANT",1)



busSourceFile = "/Users/haoni/Documents/Yelp_data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
sc = SparkContext("local", "businees data")
busData = sc.textFile(busSourceFile).cache()

revSourceFile = "/Users/haoni/Documents/Yelp_data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"  # Should be some file on your system
# revSC = SparkContext("local", "review data")
revData = sc.textFile(revSourceFile).cache()

# business_review_count = srcData.map(fetch).reduceByKey(lambda a, b: a+b).map(lambda (k,v): str(k) + ',' + str(v))
# business_review_count.saveAsTextFile('business_review_count.csv')



# business_review_user = srcData.map(fetch).groupByKey().map(convert2csv)
# business_review_user.saveAsTextFile('business_review_user.csv')


# date_count = srcData.map(fetch).reduceByKey(lambda a, b: a+b)
# print date_count.count()
# 3445
# print date_count.takeOrdered(5)
# date_count.saveAsTextFile('date_count.csv')

# business_count = srcData.map(fetch).reduceByKey(lambda a, b: a+b).count()
# print business_count

# bus_freq = srcData.map(fetchBusinessDate).reduceByKey(lambda a, b: a).map(lambda (k,v): (k[0],1)).reduceByKey(lambda a, b: a+b).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (k,v): str(k/3445.) + ',' + str(v)).map(lambda (k,v): v[0]+','+v[1])
# bus_freq.saveAsTextFile('bus_prob_.csv')

# bus_users = revData.map(fetchBusinessUserPair).reduceByKey(lambda a, b: a).map(lambda (k,v): (k[0],(k[1],1))).reduceByKey(lambda a, b: (a[0]+','+b[0],a[1]+b[1])).map(lambda (a,b): (b[1],(a,b[0]))).sortByKey(False).map(lambda (k,v): (v[0],v[1]))
bus_users = revData.map(fetchBusinessUserPair).reduceByKey(lambda a, b: a).map(lambda (k,v): (k[0],(k[1],1))).reduceByKey(lambda a, b: (a[0]+','+b[0],a[1]+b[1])).map(lambda (a,b): (b[1],(a,b[0]))).map(lambda (k,v): (v[0],(v[1],k)))


# bus_users.saveAsTextFile('bus_users_.csv')

restaurants = busData.map(fetchSpecificResturant)
restaurants.saveAsTextFile('specific_restaurants.csv')

bus_users_res = bus_users.join(restaurants).map(lambda (k,v): (v[0][1],(k,v[0][0]))).sortByKey(False).map(lambda (k,v): v[0]+','+v[1])
bus_users_res.saveAsTextFile('specific_bus_users_res.csv')

bus_features_res = bus_users.join(restaurants).map(lambda (k,v): (v[0][1],(k,v[1]))).sortByKey(False).map(lambda (k,v): v[0]+','+v[1])
bus_features_res.saveAsTextFile('specific_bus_features_res.csv')





