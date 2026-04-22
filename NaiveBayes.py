"""
Bayesian Statistics - CAS-05-601P
Naive Bayes Classification Implementation
"""

# ============================================================================
# WHAT IS NAIVE BAYES?
# ============================================================================
"""
Naive Bayes is a machine learning algorithm used for classification of tasks, 
and it is also based on the Bayes' Theorem, which calculates the probability 
of a class given certain features. It is called "naive" because it assumes 
that all the features are independent of each other, which allows it to 
compute probabilities for each class and then select the one with the highest 
likelihood. 

Also, Naive Bayes are commonly used in applications such as spam detection to 
analyze and to automatically sort or label text into categories based on context. 
This method is efficient and fast, but the main limit of it is the unrealistic 
assumption, or may encounter a probability of zero to words or data that can 
negatively affect the overall prediction.
"""

from collections import defaultdict
import math

# ============================================================================
# NAIVE BAYES CLASSIFIER IMPLEMENTATION
# ============================================================================

class SimpleNaiveBayes:
    """Simple implementation of Naive Bayes classifier"""
    
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0
    
    def train(self, data, labels):
        """Train the classifier"""
        self.total_samples = len(data)
        
        for sample, label in zip(data, labels):
            self.class_counts[label] += 1
            for feature in sample:
                self.feature_counts[label][feature] += 1
    
    def predict(self, sample):
        """Predict class for a sample"""
        scores = {}
        
        for label in self.class_counts:
            # P(Class)
            scores[label] = math.log(self.class_counts[label] / self.total_samples)
            
            # P(Features | Class)
            for feature in sample:
                count = self.feature_counts[label].get(feature, 1)
                scores[label] += math.log(count / self.class_counts[label])
        
        return max(scores, key=scores.get)


# ============================================================================
# SAMPLE SIMULATION: EMAIL SPAM DETECTION
# ============================================================================

def spam_detection_simulation():
    """Simulate Naive Bayes for email spam classification"""
    
    # Training Data: Email spam classification
    emails = [
        ['buy', 'now', 'free'],
        ['click', 'here', 'offer'],
        ['hello', 'how', 'are', 'you'],
        ['meeting', 'tomorrow', 'office']
    ]
    labels = ['spam', 'spam', 'ham', 'ham']
    
    # Train classifier
    nb = SimpleNaiveBayes()
    nb.train(emails, labels)
    
    # Test predictions
    test_emails = [
        ['buy', 'free', 'now'],      # Should predict: spam
        ['hello', 'meeting'],         # Should predict: ham
    ]
    
    print("=" * 60)
    print("NAIVE BAYES CLASSIFIER - EMAIL SPAM DETECTION")
    print("=" * 60)
    print("\nTraining Data:")
    print(f"  Spam emails: {sum(1 for l in labels if l == 'spam')}")
    print(f"  Ham emails: {sum(1 for l in labels if l == 'ham')}")
    
    print("\nTest Results:")
    print("-" * 60)
    
    for email in test_emails:
        prediction = nb.predict(email)
        print(f"Email: {email}")
        print(f"Prediction: {prediction.upper()}")
        print("-" * 60)


# ============================================================================
# SAMPLE SIMULATION 2: FRUIT CLASSIFICATION
# ============================================================================

def fruit_classification_simulation():
    """Simulate Naive Bayes for fruit classification"""
    
    # Training Data: Fruit characteristics
    fruits_data = [
        ['red', 'round', 'sweet'],      # Apple
        ['red', 'round', 'sweet'],      # Apple
        ['yellow', 'long', 'sweet'],    # Banana
        ['yellow', 'long', 'sweet'],    # Banana
        ['orange', 'round', 'sour'],    # Orange
        ['orange', 'round', 'sour']     # Orange
    ]
    fruit_labels = ['apple', 'apple', 'banana', 'banana', 'orange', 'orange']
    
    # Train classifier
    nb = SimpleNaiveBayes()
    nb.train(fruits_data, fruit_labels)
    
    # Test predictions
    test_fruits = [
        ['red', 'round', 'sweet'],      # Should predict: apple
        ['yellow', 'long', 'sweet'],    # Should predict: banana
        ['orange', 'round', 'sour']     # Should predict: orange
    ]
    
    print("\n" + "=" * 60)
    print("NAIVE BAYES CLASSIFIER - FRUIT CLASSIFICATION")
    print("=" * 60)
    print("\nTraining Data:")
    print("  Apple, Banana, Orange (2 samples each)")
    
    print("\nTest Results:")
    print("-" * 60)
    
    for fruit in test_fruits:
        prediction = nb.predict(fruit)
        print(f"Characteristics: {fruit}")
        print(f"Predicted Fruit: {prediction.upper()}")
        print("-" * 60)


# ============================================================================
# REFERENCES
# ============================================================================

REFERENCES = """
REFERENCES:
-----------
1. GeeksforGeeks - Naive Bayes Classifiers
   https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/

2. Naive Bayes Theorem
   https://en.wikipedia.org/wiki/Naive_Bayes_classifier
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run simulations
    spam_detection_simulation()
    fruit_classification_simulation()
    
    # Print references
    print(REFERENCES)