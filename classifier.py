import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# time elapsed
start_time = time.time()


# convert coordinate and label objects into 2d numpy arrays for scikit-learn
data_dict = pickle.load(open("./data.pickle", "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])


# create training and test subsets from dataset
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)


# fit model to training subset
model = RandomForestClassifier()
model.fit(x_train, y_train)


# test model with test subset, then calculate and print its accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"\n{(score*100):.2f}% of samples were classified correctly")


# dump trained model data into file for the inference classifier
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()


# confirm success and print elapsed time
print(f"\nClassifier trained and saved, time elapsed: {(time.time() - start_time):.2f}s\n")
