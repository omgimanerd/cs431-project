import cv2
import data_parser
import math


class faceDetector:
    def __init__(self, cascade):
        self.training_data = data_parser.cascadeParser().parse(cascade)
        self.cascade = self.training_data[2]
        self.pix = None
        self.pix2 = None

    def evaluate_feature(self,x,y,w,h,rects,scale):
        inv_area = 1.0/(w*h)
        sum = 0
        i = self.pix
        j = self.pix2
        total_x = i[x+w][y+h]+i[x][y]-i[x+w][y]-i[x][y+h]
        total_x2 = j[x+w][y+h]+j[x][y]-j[x+w][y]-j[x][y+h]
        norm = total_x2*inv_area-pow(total_x*inv_area,2)

        norm = math.sqrt(norm) if norm > 1 else 1

        for rect in rects:
            rx = int(scale*rect.x)
            ry = int(scale*rect.y)
            rw = int(scale*rect.width)
            rh = int(scale*rect.height)
            weight = float(rect.weight)
            sum += weight * (i[x+rx+rw][y+ry+rh]+i[x+rx][y+ry]-i[x+rx+w][y+ry]-i[x+rx][y+ry+rh])

        return sum, inv_area, norm

    def evaluate(self,x,y,w,h,scale):
        stage_pass = True
        for stage in self.training_data[0]:
            threshold = stage.threshold
            sum = 0

            for classifier in stage.weakClassifiers:
                value = 0
                i = 0

                while True:
                    feature_sum,inv_area,vnorm = self.evaluate_feature(x,y,w,h,stage[1][classifier.index],scale)

                    if feature_sum*inv_area < classifier.threshold*vnorm:
                        value = classifier.left
                        break
                    else:
                        value = classifier.right
                        break

                sum += value

            stage_pass = sum >= threshold
            if not stage_pass:
                return stage_pass

        return stage_pass

    def detectMultiScale(self, photo, scale,index,size=30,flags=None):
        results = []
        norm = cv2.normalize(photo, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        integral = cv2.integral(norm)
        window_width,window_height = size
        if flags == cv2.cv.CV_HAAR_SCALE_IMAGE:
            scale *= index+scale
        width, height = integral.shape
        self.pix = [[int((30 * photo[y, x][0] + 59 * photo[y, x][1] + 11 * photo[y, x][2]) / 100)
                    for x in range(width)] for y in range(height)]
        self.pix2 = [[self.pix[y][x] for x in range(width)] for y in range(height)]
        while window_width < width and window_height < height:
            window_width = window_height = int(scale*20)
            x = 0
            while x < height-scale*24:
                y = 0
                while y < width-scale*24:
                    if self.evaluate(x,y,window_width,window_height,scale):
                        results.append((x,y,window_width,window_height))
        return results

    def detect_face(self, photo):
        return self.cascade.detectMultiScale(photo,1.1,5)


if __name__ == "__main__":
    d = faceDetector("haarcascade_frontalface_default.xml")
    print(d.detect_face(cv2.imread('download.jpg',0)))