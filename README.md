# Spontaneous-Complex-Emotion-Classification-
Unlike the literature with six basic emotions classification, this project extends the pool of learned expression to thirteen classes. Moreover, the classified subjects are spontaneous, which is more challenging to detect. The project has concluded with slightly better accuracy than the literature with LSTM, LBP, and MFCC
Multi-modal facial emotion recognition is one of the
more challenging tasks of computer vision. Previous work has
been primarily focused on six basic emotions. This project focus
on non-basic simultaneous emotion/expression classification. We
extend the pool of learned expression to thirteen classes by the
addition of mental states such as thinking, concentrated, and etc.
We also mention results currently available in the literature and
draw comparisons with our own. To best of our knowledge, the
accuracy obtained by our models is higher than those available
in the literature.
\documentclass[journal]{IEEEtran}
\ifCLASSINFOpdf
  \usepackage[pdftex]{graphicx}
\else
\fi
\hyphenation{op-tical net-works semi-conduc-tor}
\usepackage[table,xcdraw]{xcolor}
\usepackage{graphicx}
\begin{document}

\title{Complex Facial Emotion Classification\\
\Large Project Final Report\\
\Large CS 554 - Computer Vision }




\author{{Aydamir Mirzayev,   Irmak Türköz}}

\maketitle


\begin{abstract}
Multi-modal facial emotion recognition is one of the more challenging tasks of computer vision. Previous work has been primarily focused on six basic emotions. This project focus on non-basic simultaneous emotion/expression classification. We extend the pool of learned expression to thirteen classes by the addition of mental states such as thinking, concentrated, and etc. We also mention results currently available in the literature and draw comparisons with our own. To best of our knowledge, the accuracy obtained by our models is higher than those available in the literature.
\end{abstract}

\section{Introduction}
The task of video classification is one of the more challenging tasks of computer vision. A particular sub-domain of which is the task of emotion classification. With the growing commercial interest in immediate customer satisfaction evaluation and recent advancement in deep learning, there has been a surge in the number of works addressing this problem. Similarly, this project aims to address the multi-modal emotion classification. The six facial expressions defined by Ekman and Friesen are anger, disgust, fear, joy, sadness, and surprise have been studied in many emotion classification tasks \cite{EmotionTypes}, and state-of-the-art accuracy can be larger than 0.9 for some datasets, however, in most of these studies accuracy was obtained with acted emotions and six basic emotion in a controlled environment. In this project, we assess a more challenging problem with 13 classes of emotions and spontaneous emotions, which are harder to detect because of the sudden reaction of subjects.

A significant part of the project involved cleaning and adjusting the dataset to our model. This was necessary since the dataset utilized for the project is composed of several subsets all of which needed to be transferred to a compatible format. Also due to the multi-modal nature of the project audio and video sequences of varying lengths, needed necessary adjustments. Several input configurations have been utilized. There are three types of features investigated for the classification model; facial landmarks and Local Binary Patterns(LBP) for visual data, and Mel-frequency Cepstrum (MFC) features for audio data. 
For emotion classification, a parallel learning network with late feature fusion was built. Moreover, to discuss the advantage of audio features, a separate network was trained just with the visual features in order to compare with the audio-visual feature network. 
 
\section{Dataset}
\subsection{BAUM 1s}
Dataset utilized for the project is BAUM1\cite{b1} dataset Consisting of 1,184 video clips, with 31 different subjects performing a variety of spontaneous and elicited expression, where the actors are faced towards a camera and the background is plain. Each video sequence was recorded at 29 FPS, with a resolution of (720x576) pixels, with a green screen behind the subject. Audio sequences, on the other hand, are sampled at 44KHz. Seventeen of the thirty-one subjects were female, and subjects ranged from nineteen to sixty-five years of age. Unlike the other datasets, BAUM-1s contains spontaneous reactions which are much closer to natural expressions, and are harder to detect due to shorter temporal lifespan \cite{TimeConvNets}. Moreover, the audio data is also available and there are 13 classes of emotions/expression for within dataset. These include, six basic emotions (happiness, anger, sadness, disgust, fear, surprise) and also boredom, contempt, unsure (confused, etc.), thinking, concentrating, and bothered\cite{b1}.


%\begin{figure}[htbp]
%\includegraphics[width=3.1in]{figures/baumexample.png}
%\centering
%\caption{Example frame from BAUM1s dataset}
%\label{fig:BAUMexample}
%\end{figure}



%BAUM2, on the other hand; is constructed from movie scenes which can be %considered as acted. However, one useful aspect of this dataset is the %availability of emotion scores. Therefore, instead of one class classification; %we might use the emotions with scores. Even though we are not currently using %this information, we might use this information in the final stage. Moreover, in %the current stage, we have only considered the BAUM-1 dataset as an input in %\ref{LSTM} we are using spontaneous data and in \ref{VGG16} we are using acted %data. 

\section{Feature Extraction}
\subsection{Preprossessing}
Preprocessing was required for the frames to reduce the lightning ill effect to detect the facial regions correctly with the face detection of dlib. Although we have used a pre-trained deep learning-based feature extraction method; to increase the precision and detection rate we have also implemented preprocessing step which can be considered as improvement of contrast. The preprocessing involves converting an image to lab space, apply Contrast Limited Adaptive Histogram Equalization\cite{lightning} and finally convert lab space back to RGB space.


\subsection{ OpenFace}

Initial visual features exploited in the project were facial landmark points that can detect and discriminate important facial features such as eyes, nose, mouth. We utilized OpenFace \cite{openface} face landmark extraction library for extracting 2D facial landmarks from each image of the frame. There are a total of 68 landmarks, 12 in the eye region, 9 in the nose region, 20 in the mouth region, 10 in eyebrow region, and 17 points for facial contours. To better capture head movement landmark points are centered around the position of nose tip in the first frame. Then, landmarks are normalized to [0,1] range along both dimensions. Extracted features from each frame are then reshaped into shape (L,136) where L represents the number of frames in the video sequence. 

% \begin{figure}[htbp]
% \includegraphics[width=2in]{figures/dlib-landmark-mean.png}
% \centering
% \caption{OpenFace Landmarks}
% \label{fig:openfacelandmarks}
% \end{figure}


%\par For the first model described in \ref{LSTM} we extract 68 2D facial landmark points at each frame of the video sequence. Since the video sequences in the dataset have different lengths we zero pad sequences to the maximum length. However, such an approach is far from ideal for both audio and video data, and in future work, we describe how we plan to overcome this issue.%

\subsection{ Local Binary Patterns}\label{LBP}
 
We have also experimented with second visual features, Local Binary Patterns which can give texture information of the faces. The module used for extracting LBP features is $Feature.local.binary.pattern$ from Python Scikit-Image library which is based on the circular implementation by Ojala et al. \cite{lbp}. The process of LBP extraction step-by-step is: i) extract the face region using dlib library of Python ii) divide the face region into 5x5 tiles(or 4x4 tiles in the second experiment), which makes 25 regions (16 regions in the second experiment) ii) use LBP module of scimage-feature to extract LBP image with 24 points in the circular neighborhood of radius 4. Thus, the facial image extracted from each frame is segmented into regions, and histograms for each region is obtained with the bin size of 26 to obtain 650 features (416 features for 4x4 tiles) for each frame. This procedure is illustrated in Fig. \ref{fig:lbp}.

\begin{figure}[htbp]
\includegraphics[width=3.5in]{figures/lbbp example.png}
\centering
\caption{Local Binary Pattern extraction pipeline (figure is only explanatory and does not correspond to the exact scale)}
\label{fig:lbp}
\end{figure}


\subsection{ MFCC}\label{MFCC}
For the audio features, we utilize the Mel Frequency Cepstral Coefficient (MFCC). Every audio signal in the dataset has an approximate sampling frequency of 44.1KHz, meaning, each audio signal is 44100 points in one second. We use Python Speech Features library for MFCC extraction. Due to MFCC library limitations and to decrease the dimensionality of the data, each audio feature is down-sampled by a factor of 4 to 11.025KHz and then fed to MFCC method. 

\subsection{ Windowing}\label{Windowing}
Since each video sequence is of different length, several learning approaches have been utilized. The first idea was to pad each sequence to the maximum sequence length. We have tried this approach during the first progress phase of the project. With the maximum video length around 10 seconds. however, results were less than ideal. Because long sequences required larger networks to remember inputs at the very start of the sequence, however, due to limited computational resources training with large networks was not possible and smaller networks yielded less than ideal results. 
\par Second approach was to try to train the network with variable-length data. However, this did no address the problem of memory loss for longer data sequences and slowed down the training process significantly.
\par Considering above mentioned problems it has been decided to window each video into 0.5 segments. To do this we have separated each video sequence, composed of 29 frames per second for the BAUM1s dataset, into 15 frame intervals. Corresponding audio sequence length was 22810 (before downsampling). Then labels were duplicated for each sequence section. 


%\subsubsection{Visua{Audio Data}l Data}
%\paragraph{Face Regions}
%\paragraph{Face Landmarks}
%\subsubsection
\section{Models}
We experimented with three models to discuss which features are more utilizable to do emotion recognition. In the first model, we have used audio and geometrical data, in the second model we have used audio and appearance-based data, and in the third model we have used only appearance based data.

\subsection{ Multimodal Network With Landmarks }\label{MultiModel}
For this network, we use landmarks and MFCC sequences. Each feature is fed into a separate LSTM + Dense network for initial feature extraction and then late-merged and further fed into a series of dense layers. See Fig. \ref{fig:network} for the network structure. 

\begin{figure}[htbp]
\includegraphics[width=1.4in]{network.png}
\centering
\caption{Multimodal Network}
\label{fig:network}
\end{figure}

Such a simpler network is utilized to measure the performance of simpler networks for the task and serve as a starting point for future improvements. Test dataset is obtained by splitting videos to training and test dataset in 4 to 1 ration and only then performing windowing. This ensures that during training and validation we do not perform face video recognition and avoids duplicates in the test set.  

\subsection{ Multimodal Network With LBP}\label{MultiModel2}
The second network utilized for the project is based on the first multi-modal network but differs in the fact that it utilizes Binary Local Patterns and relatively more complex architecture. MFCC features are first fed to an LSTM network and subsequently to a 2 layer dense network with batch normalization and Leaky ReLU activation. Similarly, LBP features are fed to an LSTM network followed by 3 layers of Dense network with Leaky ReLU activation. Lastly, features are late fused and fed to a triple Dense network with Softmax activation and Cross-Entropy loss. Differently from the previous network, we also perform a subject-wise split of the training and test data. That is we separate several subjects from the entire set for the test set. 

\subsection{Visual Data Only Model}\label{VisualModel}
We have developed a simpler model built on top of only, to evaluate the information gained from the audio data. The final network aimed to train with only visual data to evaluate the significance of the audio features. We have experimented with state-of-the-art pre-trained model VGG16 by feeding face images from frames directly into it. However, the results were not promising, and the network was too time-consuming. Instead, we used Local Binary Patterns described in section \ref{LBP}, to train a single LSTM model. However, since the only features are visual, we have doubled the frame rate with the windowing function that is described in section \ref{Windowing} and we have used 5x5 tiles with 26 LBP features each. Therefore, the visual data we have obtained were trained for 3352 samples with each of having feature size 30-by-650. Our model consists of an LSTM layer followed by two repeating sequels of dense, batch normalization and leaky ReLu layers and a final Dense layer (see Figure \ref{fig:LBPLSTM}). The model we built was simple and as much as close to the visual part of the multi-modal network to compare the audio information better. Our hyper-parameters are optimizer SGD with learning rate with 1-e4, loss function as categorical cross-entropy, and batch size of 64. 
\begin{figure}[htbp]
\includegraphics[width=3.3in]{figures/LSTMLBP.png}
\centering
\caption{Visual Data Only Model}
\label{fig:LBPLSTM}
\end{figure}

\section{Results}


\subsection{Multimodal Model With Landmarks}
After several experiments on the progress report with landmark features, it has been determined that face landmarks are not ideal for emotion recognition. The geometrical information given by the landmarks can also change with respect to the facial proportions of the subjects. Moreover, only the location information of these landmarks are not sufficient for training a neural network because of the low dimensional feature space. 2D landmarks often fail to capture slight appearance variations, therefore, missing the subtle expression changes. Even after optimizing the model, the maximum accuracy obtain with landmarks was 0.19. For this reason, it has been decided to use Local Binary Patterns (LBP) for the second phase of the project.
\subsection{Multimodal Model With LBPs}
As described we perform two sets of tests on the dataset. We first evaluate the performance on the previously unseen videos, and then on previously unseen subjects. Both configurations are trained for around 1000 epochs with Adagrad optimizer and a learning rate of 1-e4. 
\par For instance wise test split we obtain a maximum accuracy of  0.2875. Training is terminated at around 
1000 epochs. See Fig. \ref{fig:instancewiseaccuracy} for the plot of the learning curve of the model.


\par For subject wise test split we obtain a maximum accuracy of 0.26. We terminate the training when validation accuracy saturates. Similarly, see Fig. \ref{fig:instancewiseaccuracy} and Table \ref{tab1} for the plot of learning curve of the model and confusion matrix.

\newcommand*\rot[1]{\rotatebox{90}{#1}}
\begin{table}[h]
\centering
\caption{ Confusion Matrix For LBP Based Network with Instance-Wise split}
 \begin{tabular}{|| c c c c c c c c c c c c c||}
 \hline 
 \rot{anger} & \rot{boredom} &\rot{bothered} &\rot{concentrating} &\rot{contempt} &\rot{disgust} &\rot{fear} &\rot{happiness} &\rot{neutral} &\rot{sadness} &\rot{surprise} &\rot{thinking} &\rot{unsure}  \\
 \hline\hline
 
\cellcolor[HTML]{C0C0C0}1 & 0 & 5 & 5 & 0 & 3 & 0 & 5 & 10 & 1 & 0 & 25 & 16 \\ 
\hline
0 & \cellcolor[HTML]{C0C0C0}0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 3 & 1 \\
 \hline
0 & 0 & \cellcolor[HTML]{C0C0C0}7 & 2 & 0 & 0 & 0 & 8 & 4 & 0 & 0 & 29 & 1 \\
 \hline
0 & 0 & 0 & \cellcolor[HTML]{C0C0C0}18 & 0 & 19 & 0 & 1 & 11 & 14 & 0 & 29 & 18\\
 \hline
0 & 0 & 0 & 0 & \cellcolor[HTML]{C0C0C0}0 & 0 & 0 & 0 & 8 & 0 & 0 & 1 & 4 \\
 \hline
0 & 0 & 2 & 1 & 0 & \cellcolor[HTML]{C0C0C0}6 & 0 & 10 & 4 & 0 & 0 & 38 & 0 \\
 \hline
0 & 0 & 3 & 0 & 0 & 0 & \cellcolor[HTML]{C0C0C0}0 & 0 & 1 & 0 & 0 & 27 & 1 \\
 \hline 
0& 0&  13&  12&   0&  24&   0 &\cellcolor[HTML]{C0C0C0} 222& 37& 15& 0& 73& 19 \\
\hline
7& 0&  21&  24& 0&  8&  0& 22&  \cellcolor[HTML]{C0C0C0}76& 9&  0& 128& 51 \\
\hline
0& 0& 11& 9& 0& 15& 0& 45&  43& \cellcolor[HTML]{C0C0C0}2&  0&  47& 43 \\
\hline
0& 0& 0& 3& 0& 0&  0& 0& 0& 0& \cellcolor[HTML]{C0C0C0}0&  8&  2 \\
\hline
0& 0& 7 & 4&  0&  9& 0& 23& 51 &  2&  0&\cellcolor[HTML]{C0C0C0} 58&  19\\
\hline
1&   0& 2& 8& 0& 21& 0& 20& 35& 7& 0& 63&\cellcolor[HTML]{C0C0C0} 64 \\
\hline
\hline

\end{tabular}
\label{tab2}
\end{table}

\par As it can be observed from the confusion matrix given in Table \ref{tab2}, our model classify most of the emotions as thinking or unsure. The cause of this problem might be  suddenness reactions of natural expressions, and the videos in the dataset was not trimmed out to this sudden action. Therefore, there are many frames in the videos with thinking, or unsure faces. The temporal information of the natural expressions such as contempt, boredom was not detected due to many other frames with no actual expression. The probable reason for this the lack of information given in the dataset. 

\begin{figure}[htbp]
\includegraphics[width=3.6in]{figures/subject_instance.png}
\centering
\caption{Plot of accuracy versus epochs subject-wise validation(left) and instance-wise validation (right)}
\label{fig:instancewiseaccuracy}
\end{figure}


\begin{table*}[]
\caption{Comparison of our model with the state-of-the-art with BAUM-1s 13 classes models \cite{b1}}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{|p{12cm}|p{4cm}|}
\hline
\multicolumn{1}{|c|}{Methodology}                                                    & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}BAUM-1s Dataset Accuracy\\ (13 Classes)\end{tabular}} \\ \hline
Audio-only (by Zhalehpour et al.)                                                    & 15.29                                                                                                \\ \hline
Video-only (Zhu et al. \cite{Zhu}+ LPQ)                                              & 24.07                                                                                                \\ \hline
Video-only (Asthanea et al. \cite{CHEHRA}+ LPQ)                                      & 25.08                                                                                                \\ \hline
Video-only (Xuehan et al. \cite{IntraFace} + POEM)                                   & 23.56                                                                                                \\ \hline
\rowcolor[HTML]{EFEFEF} 
\textbf{Video-only(Ours + LBP features \ref{VisualModel})}                           & \textbf{24.34}                                                                                       \\ \hline
Audio-visual (Zhu et al. \cite{Zhu}+ LPQ)                                            & 26.18                                                                                                \\ \hline
Audio-visual (Asthanea et al. \cite{CHEHRA}+ LPQ)                                    & 25.42                                                                                                \\ \hline
Audio-visual (Xuehan et al. \cite{IntraFace} + LPQ)                                  & 25.76                                                                                                \\ \hline
\rowcolor[HTML]{EFEFEF} 
\textbf{Audio-visual (Ours + LBP and MFC features \ref{MultiModel2}) (Instancewise)} & \textbf{26.06}                                                                                       \\ \hline
\rowcolor[HTML]{EFEFEF} 
\textbf{Audio-visual (Ours + LBP and MFC features \ref{MultiModel2}) (Subjectwise)}  & \textbf{28.75}                                                                                       \\ \hline
\end{tabular}%
}
\label{tab3}
\end{table*}
\subsection{Visual Data Only Model}
One interesting experiment with the activation function in the final dense layer, the \textit{softmax} learned much faster than \textit{sigmoid}, the learning was smoother with \textit{sigmoid} instead of peak and downs. On the other hand, we have acquired 24.34\% accuracy with the \textit{softmax} with 300 batches. Considering that we are only using the LBP features of the faces, the results with respect to state-of-the-art is persuasive. Compare to the multi-modal described in \ref{MultiModel2}, we have acquired less performance with only visual data. This concludes that the audio data is informative with emotion classification.


\begin{figure}[htbp]
\includegraphics[width=3.6in]{figures/softmax.png}
\centering
\caption{Visual Only Model with LBPs Accuracy and Loss Plots}
\label{fig:softmax}
\end{figure}

\section{Discussion and Conclusion}

Many of the state-of-the-art face recognition models with BAUM -1s in the literature performed worse than the other databases used, and has been studied for \textbf{only 6 basic emotions}. Cornejo et al. has acquired 40.76 accuracy and 62.00 and 64.38 accuracies with other datasets with 6 emotions \cite{Cornejo}. Zhang et al, uses \textbf{7 emotions} from the BAUM-1s database and they get recognition accuracy 44.31 using Binaural representations with Convolutional Neural Networks \cite{Zhang}.

% Please add the following required packages to your document preamble:
% \usepackage{graphicx}
% \usepackage[table,xcdraw]{xcolor}
% If you use beamer only pass "xcolor=table" option, i.e. \documentclass[xcolor=table]{beamer}

Our literature survey concluded that emotion classification with 13 classes has not been yet proposed by other papers than the authors of BAUM-1s. Zhalehpour et al. acquired best accuracy of \textbf{25.68} with audio-visual data using IntraFace and Poem \cite{b1}. Even though we have trained and tested our system with only BAUM-1s due to time constraints since the LBP feature extraction process by each frame is time consuming, we have acquired more accuracy (\textbf{28.75}) with using multi-model with LBP's and MFCC features. The summary and comparison between the state-of-the-art is given in Table \ref{tab3}. 
 
As a conclusion, there has been many studies with basic emotions, and the topic has been considered challenging, for many classes of emotions and spontaneous emotions,it is even more challenging to propose a robust model. Although our model has better performance with the state-of-the-art, the accuracy of 13-class emotion recognition can be improved with more data, and a cleaner and more practical dataset than the BAUM-1s. We have also concluded that audio data is informative for emotion recognition when used with visual features.
%\section{Future Work}
%
%Some of the remaining problems:
%
%\begin{itemize}
%  \item An approach superior to zero paddings will be used to address the %problem of sequences with different lengths. Such approaches may include %windowing longer sequences or designing sequence length invariant networks. 
%  \item Initial multimodal architecture will be improved and optimized. %Experiments with different loss functions, learning rates and etc. will be %performed. 
%  \item GRU based networks will be tested.
%  \item We will consider other CNN based networks than VGG16. 
%  \item We will try to create a hybrid model with CNN based method (preferably %not VGG16, due to its slowness) and LSTM to combine spatial and temporal %feature learning.
%  \item We are currently using a very clean dataset, BAUM-1. However, when we %add the BAUM-2 dataset will be noisier with background, lighting effects, and %audio sound distortions.
%\end{itemize}


\ifCLASSOPTIONcaptionsoff
  \newpage
\fi

\begin{thebibliography}{1}
\bibitem{b1}S. Zhalehpour, O.Onder, Z.Akhtar and C.E.Erdem, “BAUM-1: A Spontaneous Audio-Visual Face Database of Affective and Mental States”, IEEE Trans. on Affective Computing.
\bibitem{b2}C. E. Erdem, C. Turan, Z. Aydin, "BAUM-2: A Multilingual Audio-Visual Affective Face Database", Multimedia Tools and Applications, vol. 74, No. 18, pp. 7429- 7459, 2015. 
\bibitem{lightning}"OpenCV: Histograms", Docs.opencv.org, 2020. [Online]. Available: https://docs.opencv.org/master/d6/dc7/group\_\_imgproc\_\_hist.html. [Accessed: 30- Mar- 2020].
\bibitem{openface}B. Amos, B. Ludwiczuk, M. Satyanarayanan,
"Openface: A general-purpose face recognition library with mobile applications,"
CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.
\bibitem{TimeConvNets}Lee, Jongwoong and Alexander Wong. “TimeConvNets: A Deep Time Windowed Convolution Neural Network Design for Real-time Video Facial Expression Recognition.” ArXiv abs/2003.01791 (2020).
\bibitem{EmotionTypes}P. Ekman and W. V. Friesen, "Constants across cultures in the face and emotion,"J. Pers. Soc. Psychol., vol. 17, no. 2, pp. 124–129, 1971.
\bibitem{SurveyEmotion}Imani Ma.,  Montazer G.A, "A survey of emotion recognition methods with emphasis on E-Learning environments".Journal of Network and Computer Applications, Volume 147, 2019.
\bibitem{MultiStageBinaryPatterns}Sadia A., Ayyaz H., Asim Mu., Anum N. and Sanneya A. . (2018). Multi-stage binary patterns for facial expression recognition in real world. Cluster Computing. 21.
\bibitem{lbp}T. Ojala, M. Pietikainen and T. Maenpaa, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971-987, July 2002.
\bibitem{Cornejo}J. Cornejo and H. Pedrini, "Bimodal Emotion Recognition Based on Audio and Facial Parts Using Deep Convolutional Neural Networks," 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, 2019, pp. 111-117.
\bibitem{Zhang}Zhang, Shiqing & Chen, Aihua & Guo, Wenping & Cui, Yueli & Zhao, Xiaoming & Liu, Limei. (2020). Learning Deep Binaural representations with Deep Convolutional Neural Networks for Spontaneous Speech Emotion Recognition. IEEE Access. PP. 1-1. 
\bibitem{Zhu}X. Zhu and D. Ramanan, “Face detection, pose estimation and landmark localization in the wild,” in Comput. Vis. Pattern Recog., 2012, pp. 2879–2886.
\bibitem{CHEHRA} A. Asthana, S. Zafeiriou, S. Cheng, and M. Pantic, “Incremental face alignment in the wild,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2014, pp. 1859–1866.
\bibitem{IntraFace}X. Xuehan and F. D. la Torre, “Supervised descent method and its
applications to face alignment,” in Proc. IEEE Conf. Comput. Vis.
Pattern Recog., 2013, pp. 532–539.
\end{thebibliography}

\end{document}


