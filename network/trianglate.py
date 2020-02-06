import evaluate
from PIL import Image
import json
import numpy as np
import os

ANIMALS = {0:'crocodile',1:'elephant',2:'llama',3:'snake'}

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def triangulate(bearings):

    line_params = {a:[] for a in ANIMALS.values()}

    for b in bearings:
        if len(b) > 0:
            x,y, theta = np.array(b['pose'][0:3]).T.squeeze()
            alpha = b['bearing']
            animal = b['animal']
            line_params[animal].append([x,y,theta,alpha])

    animal_coords = {}
    for animal, params in line_params.items():
        A=[]
        B=[]
        for p in params:
            x,y,theta,alpha = p
            try:
                a=np.array([-np.sin(alpha), np.cos(alpha)])@R(p[2]).T
            except Exception as e:
                import pdb; pdb.set_trace()
            A.append(a)
            b=a@np.array([x,y]).T
            B.append(b)

        A=np.array(A)
        B=np.array(B)

        animal_coords[animal]=np.linalg.lstsq(A,B,rcond=None)
    return animal_coords




















if __name__ == "__main__":
    bearings_fname = "../system_output/bearings.txt"
    folder_name = "../system_output/"
    with open(os.path.join(folder_name,bearings_fname), 'r') as f:
        bearings=json.load(f)

    animal_coords = triangulate(bearings)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # ax.scatter(z, y)

    for animal, out in animal_coords.items():
        x,y=out[0]
        ax.scatter(x, y)
        ax.annotate(animal, (x, y))
    plt.show()
    # rot = R(np.pi)


    # import pdb; pdb.set_trace()
