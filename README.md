# Compressing-Dynamical-Systems-Atari

This is both an implementation of the paper "[Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)" and an extenstion of some of their results. This paper looks at predicting future frames of the Atari video game given action sequencses. Their method works by running frames through an encoder cnn, lstm, and then decoder cnn to predict future frames. Then run those frames through the network again to predict more frames. In this github I look at using the same network but just iterating the lstm to predict future frames (figure bellow). In addition the network attempts to predict the reward of the transition between states. This makes a nice little network that acually compresses the Makov Desision process onto a small 2048 lstm. This idea is not super original and is seen in "[Embed to Control](https://arxiv.org/abs/1506.07365)" and "[Model-Based Reinforcement Learning for Playing Atari Games](http://cs231n.stanford.edu/reports2016/116_Report.pdf)".

![compressing atari onto lstm](https://github.com/loliverhennigh/Compressing-Dynamical-Systems-Atari/blob/master/figs/atari_lstm_unwrap-1.png)

## Model
There are two models that can be trained with this code. The "paper" model is a replica of the original model given in the paper. The "compression" model is very similar in structure but learns the compresses lstm representation. There is one small difference between the implementation found in the paper and the proposed models. The action vector is fed to the network before the lstm where as in the paper the action vector is fed.

## Training
You need to install the ale python atari wrapper [here](https://github.com/mgbellemare/Arcade-Learning-Environment). The training is broken up into 4 distinct parts. Seq length 1, seq length 3, seq length 5, and seq length 5 reward. The last of which attempts to predict the reward in addition to the next frame. The training is broken up this way to gradualy train on larger sequences. Each step should be trained till convergence before training on the next sequence length. To run the "paper" model run,
```
python paper_seq_1_train.py --atari_game=ms_pacman.bin
python paper_seq_3_train.py --atari_game=ms_pacman.bin
python paper_seq_5_train.py --atari_game=ms_pacman.bin
python paper_seq_5_train_reward.py --atari_game=ms_pacman.bin
```

For the "compression" model run
```
python compress_seq_1_train.py --atari_game=ms_pacman.bin
python compress_seq_3_train.py --atari_game=ms_pacman.bin
python compress_seq_5_train.py --atari_game=ms_pacman.bin
python compress_seq_5_train_reward.py --atari_game=ms_pacman.bin
```
Convergence progress can be monitored with tensorboard.


## Results


## Conclusion

This method did not work terribly well. 


