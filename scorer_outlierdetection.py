# -*- coding: utf-8 -*-

### Author: Jose Camacho Collados

import fileinput
import operator
import os
import sys
from math import sqrt


class OutlierDetectionCluster:
    # Class modeling a cluster of the dataset, composed of its topic name, its corresponding elements and the outliers to be detected
    def __init__(self, elements, outliers, topic=""):
        self.elements = elements
        self.outliers = outliers
        self.topic = topic


class OutlierDetectionDataset:
    # Class modeling a whole outlier detection dataset composed of various topics or clusters
    def __init__(self, path):
        self.path = path
        self.setWords = set()
        self.clusters = set()

    def readDataset(self):
        print("\nReading outlier detection dataset...")
        dict_cluster_elements = {}
        dict_cluster_outliers = {}
        listing = os.listdir(self.path)
        for in_file in listing:
            if in_file.endswith(".txt"):
                cluster_file = open(self.path + in_file).readlines()
                cluster_name = in_file.replace(".txt", "")
                set_elements = set()
                set_outliers = set()
                cluster_boolean = True
                for line in cluster_file:
                    if cluster_boolean:
                        if line != "\n":
                            word = line.strip().replace(" ", "_")
                            set_elements.add(word)
                            self.setWords.add(word)
                            if "_" in word:
                                for unigram in word.split("_"):
                                    self.setWords.add(unigram)
                        else:
                            cluster_boolean = False
                    else:
                        if line != "\n":
                            word = line.strip().replace(" ", "_")
                            set_outliers.add(word)
                            self.setWords.add(word)
                            if "_" in word:
                                for unigram in word.split("_"):
                                    self.setWords.add(unigram)
                self.clusters.add(
                    OutlierDetectionCluster(set_elements, set_outliers, cluster_name)
                )


def module(vector):
    # Module of a vector
    suma = 0.0
    for dimension in vector:
        suma += dimension * dimension
    return sqrt(suma)


def scalar_prod(vector1, vector2):
    # Scalar product between two vectors
    prod = 0.0
    for i in range(len(vector1)):
        dimension_1 = vector1[i]
        dimension_2 = vector2[i]
        prod += dimension_1 * dimension_2
    return prod


def cosine(vector1, vector2):
    # Cosine similarity between two vectors
    module_vector_1 = module(vector1)
    if module_vector_1 == 0.0:
        return 0.0
    module_vector_2 = module(vector2)
    if module_vector_2 == 0.0:
        return 0.0
    return scalar_prod(vector1, vector2) / (module(vector1) * module(vector2))


def pairwisesimilarities_cluster(setElementsCluster, input_vectors):
    # This function calculates all pair-wise similarities between the elements of a cluster and stores them in a dictionary
    dict_sim = {}
    for element_1 in setElementsCluster:
        for element_2 in setElementsCluster:
            if element_1 != element_2:
                dict_sim[element_1 + " " + element_2] = cosine(
                    input_vectors[element_1], input_vectors[element_2]
                )
    return dict_sim


def compose_vectors_multiword(multiword, input_vectors, dimensions):
    # Given an OOV word as input, this function either returns a vector by averaging the vectors of each token composing a multiword expression or a zero vector
    vector_multiword = [0.0] * dimensions
    cont_unigram_in_vectors = 0
    for unigram in multiword.split("_"):
        if unigram in input_vectors:
            cont_unigram_in_vectors += 1
            vector_unigram = input_vectors[unigram]
            for i in range(dimensions):
                vector_multiword[i] += vector_unigram[i]
    if cont_unigram_in_vectors > 0:
        for j in range(dimensions):
            vector_multiword[j] = vector_multiword[j] / cont_unigram_in_vectors
    return vector_multiword


def getting_vectors(path_vectors, set_words):
    # Reads input vectors file and stores the vectors of the words occurring in the dataset in a dictionary
    print("Loading word vectors...")
    dimensions = -1
    vectors = {}
    with open(path_vectors, "r", encoding="utf-8", errors="ignore") as vectors_file:
        for line in vectors_file:
            word = line.split(" ", 1)[0]
            if word in set_words:
                linesplit = line.strip().split(" ")
                if dimensions != len(linesplit) - 1:
                    if dimensions == -1:
                        dimensions = len(linesplit) - 1
                    else:
                        print("WARNING! One line with a different number of dimensions")
                vectors[word] = []
                for i in range(dimensions):
                    vectors[word].append(float(linesplit[i + 1]))
    print("Number of vector dimensions: " + str(dimensions))

    if dimensions == -1:
        # No vectors is known
        return vectors, dimensions
    for word in set_words:
        if word not in vectors and "_" in word:
            vectors[word] = compose_vectors_multiword(word, vectors, dimensions)
    print("Vectors already loaded")
    return vectors, dimensions


def main(path_dataset, path_vectors):
    print(path_dataset)
    dataset = OutlierDetectionDataset(path_dataset)
    dataset.readDataset()
    input_vectors, dimensions = getting_vectors(path_vectors, dataset.setWords)

    dict_compactness = {}
    count_total_outliers = 0
    num_outliers_detected = 0
    sum_positions_percentage = 0
    detailed_results_string = ""
    results_by_cluster_string = ""
    for cluster in dataset.clusters:
        results_by_cluster_string += "\n\n -- " + cluster.topic + " --"
        detailed_results_string += "\n\n -- " + cluster.topic + " --\n"
        num_outliers_detected_cluster = 0
        sum_positions_cluster = 0
        count_total_outliers += len(cluster.outliers)

        # Check if all vectors to the words are known
        no_vectors = []
        for elem in cluster.elements:
            if elem not in input_vectors.keys():
                no_vectors.append(elem)

        if no_vectors:
            print("Setting all test cases for cluster {} as wrongly answered, as no vector is known for:".format(cluster.topic))
            print(no_vectors)
        else:
            dictSim = pairwisesimilarities_cluster(cluster.elements, input_vectors)
            for outlier in cluster.outliers:
                if outlier not in input_vectors.keys():
                    print("Setting test case with outlier {} as wrongly answered, as no vector is known for the outlier".format(outlier))
                    continue

                comp_score_outlier = 0.0
                dict_compactness.clear()
                for element_cluster_1 in cluster.elements:
                    sim_outlier_element = cosine(
                        input_vectors[element_cluster_1], input_vectors[outlier]
                    )
                    comp_score_element = sim_outlier_element
                    comp_score_outlier += sim_outlier_element
                    for element_cluster_2 in cluster.elements:
                        if element_cluster_1 != element_cluster_2:
                            comp_score_element += dictSim[
                                element_cluster_1 + " " + element_cluster_2
                            ]
                    dict_compactness[element_cluster_1] = comp_score_element
                    detailed_results_string += (
                        "\nP-compactness "
                        + element_cluster_1
                        + " : "
                        + str(comp_score_element / len(cluster.elements))
                    )
                dict_compactness[outlier] = comp_score_outlier
                detailed_results_string += (
                    "\nP-compactness "
                    + outlier
                    + " : "
                    + str(comp_score_outlier / len(cluster.elements))
                )
                sorted_list_compactness = sorted(
                    iter(dict_compactness.items()), key=operator.itemgetter(1), reverse=True
                )
                position = 0
                for element_score in sorted_list_compactness:
                    element = element_score[0]
                    if element == outlier:
                        sum_positions_cluster += position
                        if position == len(cluster.elements):
                            num_outliers_detected_cluster += 1
                        break
                    position += 1
                detailed_results_string += (
                    "\nPosition outlier "
                    + outlier
                    + " : "
                    + str(position)
                    + "/"
                    + str(len(cluster.elements))
                    + "\n"
                )

        num_outliers_detected += num_outliers_detected_cluster
        sum_positions_percentage += (sum_positions_cluster * 1.0) / len(
            cluster.elements
        )
        scoreOPP_cluster = (
            ((sum_positions_cluster * 1.0) / len(cluster.elements))
            / len(cluster.outliers)
        ) * 100
        accuracy_cluster = (
            (num_outliers_detected_cluster * 1.0) / len(cluster.outliers)
        ) * 100.0
        results_by_cluster_string += "\nAverage outlier position in this topic: " + str(
            scoreOPP_cluster
        )
        results_by_cluster_string += (
            "\nOutliers detected percentage in this topic: " + str(accuracy_cluster)
        )
        results_by_cluster_string += "\nNumber of outliers in this topic: " + str(
            len(cluster.outliers)
        )

    scoreOPP = ((sum_positions_percentage * 1.0) / count_total_outliers) * 100
    accuracy = ((num_outliers_detected * 1.0) / count_total_outliers) * 100.0
    print("\n\n ---- OVERALL RESULTS ----\n")
    print("OPP score: " + str(scoreOPP))
    print("Accuracy: " + str(accuracy))
    print("\nTotal number of outliers: " + str(count_total_outliers))

    print("Would you like to see the results by topic?")
    print(results_by_cluster_string)
    print(detailed_results_string)


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) == 2:

        path_dataset = args[0]
        path_vectors = args[1]

        main(path_dataset, path_vectors)

    else:
        sys.exit(
            """
            Requires:
            path_dataset -> Path of the outlier detection directory
            path_vectors -> Path of the input word vectors
            """
        )
