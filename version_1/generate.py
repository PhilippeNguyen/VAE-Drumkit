print('loading...')
import argparse
import keras
import os
import keras.backend as K
import numpy as np
import tensorflow as tf
import sys
import librosa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import sounddevice as sd
from tkinter import filedialog

fs = os.path.sep
extra_layers_path = os.path.realpath(__file__).split(fs)[:-2]
sys.path.append(fs.join(extra_layers_path))
from extra_layers import NNResize,Spectrogram

class MainApp(tk.Frame):
    def __init__(self,root,model):
        super(MainApp, self).__init__()
        self.root = root
        self.model = model
        
        ##############
        ## Settings ##
        ##############
        self.sr=22050
        self.N_FFT=2048
        self.HOP_LEN = 512
        self.expected_len = 22050
        self.front_padding = 4410
        self.to_amp = lambda x : np.exp(x) - 1
        self.dim_latent = 8
        self.build_generate(model)
        
        self.audio = np.zeros(self.expected_len)
        
        
        ############
        ## Latent ##
        ############
        #Sliders

        self.coord_label = tk.Label(self,text="Coordinates")
        self.coord_label.pack()
        self.coord_text = [None]*self.dim_latent
        self.coord_sliders = [None]*self.dim_latent
        self.current_coords = [0]*self.dim_latent
        for idx in range(8):
            self.coord_text[idx] = tk.Label(self,text=str(idx))
            self.coord_text[idx].pack()
            self.coord_sliders[idx] = tk.Scale(self, from_=-3, to=3, 
                                             resolution=0.05, length = 225,
                                             orient=tk.HORIZONTAL)
            self.coord_sliders[idx].pack()

        #Button
        
        self.applyButton = tk.Button(self, text = 'Generate', width = 25, 
                    command = self.generate)
        self.applyButton.pack() 
        
        ####################
        ## Saving/Viewing ##
        ####################
        
        #Volume
        self.volume_text = tk.Label(self,text="Volume")
        self.volume_text.pack()
        self.volume_slider = tk.Scale(self, from_=0, to=3, 
                                         resolution=0.05, length = 225,
                                         orient=tk.HORIZONTAL)
        self.volume_slider.set(1)
        self.volume_slider.pack()
        
        self.playButton = tk.Button(self, text = 'Play', width = 25, 
            command = self.play)
        self.playButton.pack() 
        self.saveButton = tk.Button(self, text = 'Save', width = 25, 
            command = self.save)
        self.saveButton.pack() 
        
        
        ############
        ## Canvas ##
        ############
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.plot(self.audio)
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()
        
        print('loaded')
        self.pack(fill=tk.BOTH,expand=1)
        self.focus_set()
    
    def get_audio(self):
        volume = self.volume_slider.get()
        return volume*self.audio
    def play(self):
        audio = self.get_audio()
        sd.play(audio,samplerate=self.sr)
    def save(self):
        savefile = filedialog.asksaveasfilename(defaultextension='.wav',
                                                filetypes =(("wav files","*.wav"),
                                                            ("all files","*.*")))
        librosa.output.write_wav(savefile,self.get_audio(),sr=self.sr)
    def build_generate(self,model):
        
        #################
        ## Model Setup ##
        #################
        
        sess = K.get_session()
        
        model = keras.models.load_model(model,compile=False,
                                        custom_objects={'NNResize':NNResize})
        
        encoder_input = model.input
        encoder_output = model.get_layer('z_mean').output
        encoder = K.Function([encoder_input],[encoder_output])
        
        decoder_input = model.get_layer('dense_1').input
        decoder_output = model.output
        decoder = K.Function([decoder_input],[decoder_output])
        
        output_shape = decoder_output.get_shape().as_list()
        
        ##########################
        ## Function Definitions ##
        ##########################
        
        tf_spectrogram = Spectrogram(n_fft=self.N_FFT,
                                     hop_length=self.HOP_LEN,freq_format='freq_last')
    
    
        mag_placeholder = tf.placeholder(shape=(1,*output_shape[1:]),dtype=np.float32)
        
        alpha = 100
        init_recon = np.random.randn(1,int(self.expected_len)).astype(np.float32)
        signal_len = init_recon.shape[1]
        recon = tf.Variable(init_recon)
        
        recon_mel_out = tf_spectrogram.call(recon)
        stft_tf = tf.contrib.signal.stft(recon,frame_length=self.N_FFT,
                                         frame_step=self.HOP_LEN,pad_end=True)
        x_tf = tf.contrib.signal.inverse_stft(stft_tf,frame_length=self.N_FFT,
                                              frame_step=self.HOP_LEN,
                                              fft_length=self.N_FFT)[:,:signal_len]
        
        x_loss = tf.reduce_sum(tf.square(recon-x_tf))
        mag_loss = tf.reduce_sum(tf.square(mag_placeholder-recon_mel_out))
        
        recon_loss = alpha*x_loss + mag_loss
        sess.run(recon.initializer)
        
        recon_opt = tf.contrib.opt.ScipyOptimizerInterface(
              recon_loss, method='L-BFGS-B', options={'maxiter': 500},
              var_list=[recon])
        
        
        def generate():
            for idx,coord_slider in enumerate(self.coord_sliders):
                self.current_coords[idx] = coord_slider.get()
                
            decoder_input = np.asarray([self.current_coords],dtype=np.float32)
            decoder_output = decoder([decoder_input])[0]
            amp_out = self.to_amp(decoder_output)
            amp_out[:,:(self.front_padding//self.HOP_LEN)] = 0
        
            feed_dict={mag_placeholder:amp_out}
            recon_opt.minimize(sess,feed_dict=feed_dict)
            print('Recon loss:', recon_loss.eval(session=sess,feed_dict=feed_dict))
            recon_out = recon.eval(session=sess)
            
            self.audio =  recon_out[0]
            
            self.ax.clear()
            self.ax.plot(self.audio)
            self.canvas.draw()
            
        self.generate = generate
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', action='store', dest='model',
                    required=True,
        help='path to the version 1 model')

    
    args = parser.parse_args()
    
    root = tk.Tk()
    MainApp(root,args.model).pack(side="top", fill="both", expand=True)
    root.mainloop()