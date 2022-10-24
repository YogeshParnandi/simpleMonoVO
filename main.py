
import cv2
import numpy as np
import os

FIRST_FRAME = 1
DEFAULT = 0

camera = { "width": 1241, "height": 376, "F": 718, "PP": (607, 185) ,"K" : np.array([[718, 0, 607], [0,718,185], [0,0,1]])}

class odom:
    def __init__(self, poses):
        self.frameStage = FIRST_FRAME
        self.currentFrame = None
        self.previousFrame = None
        self.detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.descriptor = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
        self.currentTranslation = np.array([[0], [0], [0]])
        self.currentRotation = np.identity(3)
        self.trueX, self.trueY, self.trueZ = 0,0,0
        with open(poses) as f:
            self.poses = f.readlines()
        self.initialised = False
    
    def getAbsoluteScale(self, frame_id):  
        # specialized for KITTI odometry dataset
        ss = self.poses[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.poses[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
    
    def estimatePose(self, prevImage, currImage, frameID):
        pKP, prevDesc = self.detector.detectAndCompute(prevImage, None)
        cKP, currDesc = self.detector.detectAndCompute(currImage, None)
        matches_raw = self.matcher.match(prevDesc, currDesc)
        matches = []
        prevKP= []
        currKP = []
        min_dist = 10000
        max_dist = 0

        for i in range(np.shape(prevDesc)[0]):
            dist = matches_raw[i].distance
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

        for i in range(np.shape(prevDesc)[0]):
            if matches_raw[i].distance <= max(2 * min_dist, 30):
                matches.append(matches_raw[i])
        
        for i in range(len(matches)):
            prevKP.append(pKP[matches[i].queryIdx].pt)
            currKP.append(cKP[matches[i].trainIdx].pt)
            
        prevKP = np.array(prevKP)
        currKP = np.array(currKP)
        matches = np.array(matches)

        if not self.initialised:
            E, mask = cv2.findEssentialMat(currKP, prevKP, cameraMatrix=camera["K"], method=cv2.RANSAC, prob=0.999, threshold=1)
            retval, self.currentRotation, self.currentTranslation, mask = cv2.recoverPose(E, currKP, prevKP, cameraMatrix=camera["K"])
            self.initialised = True
        else:
            E, mask = cv2.findEssentialMat(currKP, prevKP, cameraMatrix=camera["K"], method=cv2.RANSAC, prob=0.999, threshold=1)
            retval, R, t, mask = cv2.recoverPose(E, currKP, prevKP, cameraMatrix=camera["K"])
            absolute_scale = self.getAbsoluteScale(frameID)
            if(absolute_scale > 0.1):
                self.currentTranslation = self.currentTranslation + absolute_scale * self.currentRotation.dot(t)
                self.currentRotation = R.dot(self.currentRotation)

    def updatePose(self, img, frameID):
        self.currentFrame = img
        if self.frameStage == FIRST_FRAME:
            self.frameStage = DEFAULT
        elif self.frameStage == DEFAULT:
            self.estimatePose(self.previousFrame, self.currentFrame, frameID)
        self.previousFrame = self.currentFrame


if __name__ == '__main__':
    print("hello VO")
    N = 4541
    traj = np.zeros((768,1024,3), dtype=np.uint8)
    # iterate through images
    odom_ = odom('/xxx/00.txt')
    for i in range(1, N):
        imagePath = '/xxx/data_odometry_gray/dataset/sequences/00/image_0/' + str(i).zfill(6) + '.png'
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        odom_.updatePose(image, i)
        cur_t = odom_.currentTranslation

        # Plot function from "https://github.com/uoip/monoVO-python"
        if(i > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x)+290, int(z)+90
        true_x, true_y = int(odom_.trueX)+290, int(odom_.trueZ)+90
        cv2.circle(traj, (draw_x,draw_y), 1, (i*255/4540,255-i*255/4540,0), 1)
        cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        # cv2.imshow('camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)
cv2.imwrite('map.png', traj)