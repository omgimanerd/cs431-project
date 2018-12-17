"""
File: data_parser.py
"""

import xml.etree.ElementTree as ET
import cv2

class stage():
    def __init__(self, index, threshold, max_weak_count):
        self.weakClassifiers = []
        self.index = index
        self.threshold = threshold
        self.max_weak_count = max_weak_count

class weakClassifier():
    def __init__(self,left,right,index,threshold):
        self.left = left
        self.right = right
        self.index = index
        self.threshold = threshold
    """
    Returns the threshold needed for the given feature value for this particular classifier
    """
    def check_threshold(self, threshold):
        return self.left if threshold < self.threshold else self.right

class feature():
    def __init__(self, index, rects):
        self.index = index
        self.rects = rects


class rect():
    def __init__(self, x,y,width,height,weight):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.weight = weight


class cascadeParser:

    """
    Returns a tuple of stages and features
    """
    def parse(self,file_string):
        f = open(file_string)
        tree = ET.parse(f)

        root = tree.getroot()

        stage_list = []
        i = 0
        for stages in root.iter("stages"):
            for stage_e in stages.iter("_"):
                # Create stage
                n_stage = None
                threshold = 0
                count = 0
                try:
                    threshold = float(stage_e.find("stageThreshold").text)
                    count = int(stage_e.find("maxWeakCount").text)
                    n_stage = stage(i,threshold,count)
                except:
                    n_stage = stage(i, None, None)
                i += 1
                # Get classifiers in stage
                for w_classifiers in stages.iter('weakClassifiers'): # .findall('./cascade/stages'):
                    for classifier in w_classifiers:

                        # Get internal nodes
                        internal_nodes = classifier.find("internalNodes").text.split(' ')
                        index = int(internal_nodes[14]) # there's a bunch of whitespace
                        threshold = float(internal_nodes[15])

                        # Get leaf values
                        leaf_values = classifier.find("leafValues").text.split(' ')
                        left = leaf_values[12]
                        right = leaf_values[13]

                        # Add classifier to stage
                        w_classifier = weakClassifier(left,right,index,threshold)
                        n_stage.weakClassifiers.append(w_classifier)
                stage_list.append(n_stage)
        verify = cv2.CascadeClassifier(file_string)

        feature_list = []
        i = 0
        for features in root.iter("features"):
            for feature_e in features.iter("_"):
                # Create feature

                # Get classifiers in stage
                rects_list = []
                for rects in feature_e.iter('rects'):  # .findall('./cascade/stages'):
                    for rect_e in rects.iter('_'):
                        rect_t = rect_e.text.split(' ')
                        # print(rect_t)
                        rects_list.append(rect(int(rect_t[10]),int(rect_t[11]),
                                              int(rect_t[12]),int(rect_t[13]),
                                              float(rect_t[14]))) # x,y,w,h,weight

                feature_list.append(feature(i,rects_list))
                i += 1

        return stage_list, feature_list, verify




if __name__ == "__main__":
    parser = cascadeParser()
    data = parser.parse("haarcascade_frontalface_default.xml")
    print(data[0]," ",data[1])