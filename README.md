**Different fake data generation approaches:**
- 2 projects are from the Face-Xray paper
- Face-Xray-master uses a few more processing steps
- third project, CVPRW2019_Face_Artifacts-master:
    - they do a few things differently

*** Face-Xray approach: ***
Given: images of faces, in best case same image sizes and somehow centered

Goal: Given two images, select face region from first one, do some transformations to the select face mask and paste it to second image

Steps: paste face B to A
1. get landmarks from Face A
2. Search for most similar face in given data collection --> face B
3. from landmarks from A, get hull --> mask (region to extract pixels from)
4. deform the mask (piecwise affine transform, blurr, etc.)
5. color correct image B to A
7. since our images are as similar as possible given the landmarks: use deformed mask from A to extract pixels from B and paste to A (faceswap)

The more complex paper has additional steps:
8. down and up sample again
9. use jpeg comrpession

The more complex paper also experiments with different Mask formation algorithms, given the facial landmarks

See Notebooks for reference

*** Face-Artifacts approach: ***

Similar transformation and mask generation etc.

But instead of step 2. from before (searching for most similar face), the landmarks of ALL faces are transformed into a normalized space (such that all images look similar)
--> I don't know how well this approach works, I still need to test this


*** Requirements ***
See requirements file here
There may be some additional packages not needed, but this current environment was created only for the project. So this should be as slim as possible.


Requirements for Face-Artifacts will be created