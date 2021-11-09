import numpy as np
from stickman import stick_man_generator

def generate_evaluation_dataset(batch_size,nb_batch,args):

    val_set = stick_man_generator(
        batch_size = batch_size)
        #set_of_data = 'val', 
        #p_circles=args.circles, 
        #p_squares=args.squares, 
        #p_real=args.real, 
        #input_shape = (args.shape,args.shape,3))

    data = val_set.data_generation(nb_batch)
    X = data[1]
    y = data[0]
    np.save("../img_validation",X)
    np.save("../kp_validation",y)

def load_validation():
    img = np.load("../img_validation.npy")
    kp = np.load("../kp_validation.npy")

    return img, kp

if __name__ == "__main__":
    generate_evaluation_dataset(128,50,0)