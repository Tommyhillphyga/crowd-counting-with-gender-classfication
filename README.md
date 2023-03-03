# A Crowd Counting Application
### About 
This app uses a lightweight face-detector to detect regions containing faces from image and video streams and give a total count of the people present.
It combines gender classifier with the face detector to categorize people and give an accurate count of the people present in a video stream.

### Run Demo
- Clone repo.
- Install requiremets
```
pip install -r requiremnts.txt
```
- Run the following command:
```
python detection.py --weights weights/weights.pt --input sample2.mp4 --output results.mp4
```

