import tensorflow as tf
import numpy as np
import scipy
import external.poissonblending as blending
from scipy.signal import convolve2d
import Parser
import os
from scipy.misc import toimage


class ModelInpaint():
    def __init__(self, modelfilename, config,
                 model_name='dcgan',
                 gen_input='z:0', gen_output='Tanh:0', gen_loss='Mean_2:0',
                 disc_input='real_images:0', disc_output='Sigmoid:0',
                 z_dim=100, batch_size=64):
        """
        Model for Semantic image inpainting.
        Loads frozen weights of a GAN and create the graph according to the
        loss function as described in paper

        Arguments:
            modelfilename - tensorflow .pb file with weights to be loaded
            config - training parameters: lambda_p, nIter
            gen_input - node name for generator input
            gen_output - node name for generator output
            disc_input - node name for discriminator input
            disc_output - node name for discriminator output
            z_dim - latent space dimension of GAN
            batch_size - training batch size
        """

        self.config = config

        self.batch_size = batch_size
        self.z_dim = z_dim
        self.graph, self.graph_def = ModelInpaint.loadpb(modelfilename,
                                                         model_name)

        self.gi = self.graph.get_tensor_by_name(model_name + '/' + gen_input)
        self.go = self.graph.get_tensor_by_name(model_name + '/' + gen_output)
        self.gl = self.graph.get_tensor_by_name(model_name + '/' + gen_loss)
        self.di = self.graph.get_tensor_by_name(model_name + '/' + disc_input)
        self.do = self.graph.get_tensor_by_name(model_name + '/' + disc_output)

        self.image_shape = self.go.shape[1:].as_list()

        self.l = config.lambda_p
        self.l2 = 0.1
        self.parsing_loss=0

        self.sess = tf.Session(graph=self.graph)

        self.init_z()

    def init_z(self):
        """Initializes latent variable z"""
        self.z = np.random.randn(self.batch_size, self.z_dim)

    def sample(self, z=None):
        """GAN sampler. Useful for checking if the GAN was loaded correctly"""
        if z is None:
            z = self.z
        sample_out = self.sess.run(self.go, feed_dict={self.gi: z})
        return sample_out

    def preprocess(self, images, imask, useWeightedMask=True, nsize=7):
        """Default preprocessing pipeline
        Prepare the data to be fed to the network. Weighted mask is computed
        and images and masks are duplicated to fill the batch.

        Arguments:
            image - input image
            mask - input mask

        Returns:
            None
        """
        images = ModelInpaint.imtransform(images)

        if useWeightedMask:
            mask = ModelInpaint.createWeightedMask(imask, nsize)
        else:
            mask = imask

        mask = ModelInpaint.create3ChannelMask(mask)

        bin_mask = ModelInpaint.binarizeMask(imask, dtype='uint8')
        self.bin_mask = ModelInpaint.create3ChannelMask(bin_mask)

        self.masks_data = np.repeat(mask[np.newaxis, :, :, :],
                                    self.batch_size,
                                    axis=0)

        # Generate multiple candidates for completion if single image is given
        if len(images.shape) is 3:
            self.images_data = np.repeat(images[np.newaxis, :, :, :],
                                         self.batch_size,
                                         axis=0)
        elif len(images.shape) is 4:
            # Ensure batch is filled
            num_images = images.shape[0]
            self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                         self.batch_size,
                                         axis=0)
            ncpy = min(num_images, self.batch_size)
            self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()

    def postprocess(self, g_out, blend=True):
        """Default post processing pipeline
        Applies poisson blending using binary mask. (default)

        Arguments:
            g_out - generator output
            blend - Use poisson blending (True) or alpha blending (False)
        """
        images_out = ModelInpaint.iminvtransform(g_out)
        images_in = ModelInpaint.iminvtransform(self.images_data)

        if blend:
            for i in range(len(g_out)):
                images_out[i] = ModelInpaint.poissonblending(
                    images_in[i], images_out[i], self.bin_mask
                )
        else:
            images_out = np.multiply(images_out, 1 - self.masks_data) \
                         + np.multiply(images_in, self.masks_data)

        return images_out

    def build_inpaint_graph(self):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
            self.masks = tf.placeholder(tf.float32,
                                        [None] + self.image_shape,
                                        name='mask')
            self.images = tf.placeholder(tf.float32,
                                         [None] + self.image_shape,
                                         name='images')
            self.context_loss = tf.reduce_sum(
                tf.contrib.layers.flatten(
                    tf.abs(tf.multiply(self.masks, self.go) -
                           tf.multiply(self.masks, self.images))), 1
            )

            self.ploss = tf.placeholder(tf.float32, name = 'ploss')

            # ---------------------------------------------------------------------------------------------
            # # Get all input images and generated images
            #
            # # GET coordinates and find coordinate difference and calculate parse loss
            self.perceptual_loss = self.gl
            self.inpaint_loss = self.context_loss + self.l * self.perceptual_loss + self.l2 * self.ploss
            self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)

    def getParseLoss(self, in_x, in_y, gen_img):
        loss = 0

        #Run loop for all image in batch
        for i in range(64):
            #Get the landmark coordinates of the masked generated image
            gen_masked = gen_img[i, :, :, :]
            gen_masked = scipy.misc.toimage(gen_masked)
            gen_masked = np.array(gen_masked)
            gen_coor = Parser.getShape(gen_masked)

            gen_x = gen_coor[:, 0]
            # print('gen_x:',gen_x)

            gen_y = gen_coor[:, 1]
            # print('gen_y:',gen_y)

            for j in range(68):
                if gen_x[j] < in_x[j] - 3 or gen_x[j] > in_x[j] + 3:
                    loss = loss + 0.5
                else:
                    loss = loss + 0
                if gen_y[j] < in_y[j] - 3 or gen_y[j] > in_y[j] + 3:
                    loss = loss + 0.5
                else:
                    loss = loss + 0

        return loss

    def inpaint(self, image, mask, blend=True):
        """Perform inpainting with the given image and mask with the standard
                pipeline as described in paper. To skip steps or try other pre/post
                processing, the methods can be called seperately.

                Arguments:
                    image - input 3 channel images
                    mask - input binary mask, single channel. Nonzeros values are
                           treated as 1
                    blend - Flag to apply Poisson blending on output, Default = True

                Returns:
                    post processed image (merged/blneded), raw generator output
                """

        self.preprocess(image, mask)

        self.build_inpaint_graph()

        imout = self.backprop_to_input()

        return self.postprocess(imout, blend), imout

    def backprop_to_input(self, verbose=True):
        """Main worker function. To be called after all initilization is done.
        Performs backpropagation to input using (accelerated) gradient descent
        to obtain latent space representation of target image

        Returns:
            generator output image
        """
        v = 0

        #Input image is masked, processesd to be readable by opencv
        #and landmark coordinates obtained as in_x and in_y
        parse_input_image = self.images_data * self.masks_data
        self.saveimages(parse_input_image, 'parse_in', 'in')
        parse_input_image = parse_input_image[0, :, :, :]
        parse_input_image = scipy.misc.toimage(parse_input_image)
        parse_input_image = np.array(parse_input_image)
        parse_input_image = Parser.getShape(parse_input_image)
        in_x = parse_input_image[:, 0]
        # print('in_x:',in_x)
        in_y = parse_input_image[:, 1]
        # print('in_y:',in_y)


        for i in range(self.config.nIter):

            # Genrated images are just masked.
            parse_gen_image = self.sample() * self.masks_data

            #Calculate parsing loss, provided masked and processed input
            #image along with (only) masked generated image.
            ploss = self.getParseLoss(in_x, in_y ,parse_gen_image)

            #----------------------------------------------------------------------------------

            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go]
            in_dict = {self.masks: self.masks_data,
                       self.gi: self.z,
                       self.images: self.images_data,
                       self.ploss : ploss
                       }

            loss, grad, imout = self.sess.run(out_vars, feed_dict=in_dict)

            v_prev = np.copy(v)
            v = self.config.momentum * v - self.config.lr * grad[0]
            # Traversal in latent space
            self.z += (-self.config.momentum * v_prev +
                       (1 + self.config.momentum) * v)
            self.z = np.clip(self.z, -1, 1)

            blend=False

            if (i%10==0):
                t_inpaint=self.postprocess(imout, blend)

                #Save inpainted image sample after every 10 iteration
                self.saveimages(t_inpaint, 'samples', 'inpaint')

                #Save generated image sample after every 10 iteration
                self.saveimages(parse_gen_image, 'parse_gen', 'gen')

            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))

        return imout

    @staticmethod
    def loadpb(filename, model_name='dcgan'):
        """Loads pretrained graph from ProtoBuf file

        Arguments:
            filename - path to ProtoBuf graph definition
            model_name - prefix to assign to loaded graph node names

        Returns:
            graph, graph_def - as per Tensorflow definitions
        """
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                op_dict=None,
                                producer_op_list=None,
                                name=model_name)

        return graph, graph_def

    @staticmethod
    def imtransform(img):
        """Helper: Rescale pixel value ranges to -1 and 1"""
        return np.array(img) / 127.5 - 1

    @staticmethod
    def iminvtransform(img):
        """Helper: Rescale pixel value ranges to 0 and 1"""
        return (np.array(img) + 1.0) / 2.0

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blending.blend(img1, img2, 1 - mask)

    @staticmethod
    def createWeightedMask(mask, nsize=7):
        """Takes binary weighted mask to create weighted mask as described in
        paper.

        Arguments:
            mask - binary mask input. numpy float32 array
            nsize - pixel neighbourhood size. default = 7
        """
        ker = np.ones((nsize, nsize), dtype=np.float32)
        ker = ker / np.sum(ker)
        wmask = mask * convolve2d(mask, ker, mode='same', boundary='symm')
        return wmask

    @staticmethod
    def  binarizeMask(mask, dtype=np.float32):
        """Helper function, ensures mask is 0/1 or 0/255 and single channel
        If dtype specified as float32 (default), output mask will be 0, 1
        if required dtype is uint8, output mask will be 0, 255
        """
        assert (np.dtype(dtype) == np.float32 or np.dtype(dtype) == np.uint8)
        bmask = np.array(mask, dtype=np.float32)
        bmask[bmask > 0] = 1.0
        bmask[bmask <= 0] = 0
        if dtype == np.uint8:
            bmask = np.array(bmask * 255, dtype=np.uint8)
        return bmask

    @staticmethod
    def create3ChannelMask(mask):
        """Helper function, repeats single channel mask to 3 channels"""
        assert (len(mask.shape) == 2)
        return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    def saveimages(self, outimages, directory, prefix='samples'):
        numimages = len(outimages)
        for i in range(numimages):
            filename = '{}_{}.png'.format(prefix, i)
            filename = os.path.join(directory, filename)
            # outimages[a,b,c,d] means a=image no., b=height, c=width, d=colorchannels
            scipy.misc.imsave(filename, outimages[i, :, :, :])
