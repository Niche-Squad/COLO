# Materials and Methods

## Cow Husbandry

The studied cows were housed in a free-stall barn at Virginia Tech Dairy Comlex at Kentland Farm in Virginia, USA. The cow handling and image capturing were conducted following the guidelines and approval of the Virginia Tech Institutional Animal Care and Use Committee (#IACUC xxxxx).

...

## Image Collection

The studied images were collected using Amazon Ring camera model Spotlight Cam Battery Pro (Ring Inc.), which provides a real-time video stream of the dairy cows. There were three cameras being installed in the barn: Two cameras were set at the same height of 3.25 meters (10.66 feet) above the ground, capturing the same area of 33.04 square meters (355.67 square feet) in the barn. One of them faced the cows from the top view, while the other was angled downward around 40 degrees from the horizontal plane to capture the side view of the cows. Hereafter the first camera is referred to as  \textit{the top-view camera}, and the second camera is referred to as \textit{the side-view camera}. Additionally, to create an external dataset for examining the model generalization, a third camera was installed at a lower height of 2.74 meters (9.00 feet) capturing a different area of 77.63 suqare meters (835.56 square feet) in the barn. The third camera was angled 10 degree downward from the horizontal plane, providing a more challenging perspective where cows are more likely to occlude each other. This camera hereafter is referred to as \textit{the external camera}.
Images were captured using an unofficial Ring Application Programming Interface (API) \citep{greif_dgreifring_2024} that allows automating the process of capturing snapshots from the video stream.

Our study starts with the systematic acquisition of image data, focusing on targeted cattle populations within Kentland Farm at Virginia Tech.

All animal handling and media recordings were conducted following the guidelines and approval of the Virginia Tech Institutional Animal Care and Use Committee. This initial phase is succeeded by meticulous data processing steps which include the annotation and formatting of the dataset for machine learning applications. Subsequently, we proceed to fine-tune our dataset utilizing a variety of deep learning architectures. For each model, we meticulously calculate a suite of performance metrics.

Building upon the results obtained from these diverse models, we construct a "summary plot." This plot is designed to elucidate the findings related to the second and third questions delineated in Section {contribution} of our paper. It will visually guide the selection of an optimal model by delineating the relationship between dataset size and achieved accuracy, as well as the computational cost versus the precision of the models. Through this analytical representation, we aim to furnish a comprehensive tool that aids researchers in making informed decisions when it comes to choosing the most suitable object detection model for their specific requirements in livestock production studies.

\subsection*{Data Preparation}

Recognizing the crucial role of lighting conditions on data integrity, we meticulously orchestrated our data gathering operations at assorted intervals throughout the day, specifically: dawn, midday, dusk, and late evening. This methodical approach was paramount in guaranteeing the inclusion of an extensive spectrum of lighting conditions within our dataset, thereby augmenting its diversity and resilience to various environmental challenges.

Moreover, cognizant of the effect camera angles and perspectives have on capturing the full gamut of cattle postures, we varied our image capture process accordingly. This variation not only accounted for the different positions and movements of the cattle but also for the heterogeneous nature of the environment in which they were situated. In addition, we aimed to ensure a broad representation of breeds by including both Jersey and Holstein cows in our dataset, recognizing that breed-specific characteristics could significantly influence the model's performance.

\subsubsection*{Animal Husbandry}

The farm, hosting a diverse bovine community of over 200 individuals from the Jersey and Holstein breeds, served as an exemplary setting for our endeavor. It offered a plethora of varied scenarios and animal interactions, encapsulating the essence of a vibrant and dynamic agricultural environment. This multifaceted setting was critical in establishing a robust and comprehensive dataset reflective of the real-world complexity and variability one would expect in a livestock farming operation.

\subsubsection*{Camera Setup}

The cornerstone of our data acquisition process was the deployment of Amazon Ring Cameras, as illustrated in Figure ~\ref{fig:camera-roboflow}. These cameras, primarily acclaimed for their real-time video surveillance capabilities, were judiciously selected for their advanced functionalities that are particularly conducive to farm monitoring applications. The Amazon Ring cameras are engineered to deliver 1080HD video quality, equipped with infrared night vision for after-dark monitoring, and Live View features for real-time observation.

Our rationale behind this choice stems from the device's capacity to provide high-definition 1080p video quality, ensuring that the clarity of the footage is not compromised, which is crucial for the accuracy of object detection algorithms. The integrated infrared night vision capability ensures continuous operation, day and night, which is imperative for creating a dataset that reflects all possible environmental conditions encountered in a farm setting.

Moreover, the Ring camera's battery-operated design introduces a level of convenience and adaptability that is ideally suited for the agricultural context. With a rechargeable battery that can last approximately one month per charge, the system provides a sustained, maintenance-low operation. This feature is particularly advantageous in farm environments where power sources may not be readily available at various points of interest.

Accessibility and manageability of the camera system are further enhanced through its compatibility with mobile and computer-based applications, allowing for remote access and control. Users receive prompt notifications on the status of the battery, thus ensuring that the camera's operation remains uninterrupted through proactive maintenance alerts for battery recharging or replacement. This negates the need for a permanent wired infrastructure and embodies a synergetic combination of high-resolution imaging capabilities with operational dexterity, making it an excellent tool for comprehensive monitoring and data collection in dynamic farm environments.

We used two Amazon Ring cameras for a dual-view approach. We positioned one camera to provide a top-down perspective, while the other was configured for lateral views. This bimodal configuration was designed to grasp the full geometric profile of the cattle, an approach that is conducive to enhancing the object detection algorithms' ability to discern the cows with greater precision.

The top view camera is critical in capturing the distinctive outlines and patterns of the cows' backs, which often include unique color markings and spine curvature, valuable features for individual identification and count. The side view camera, on the other hand, captures the profile shapes, including height, length, and body condition, offering a different set of attributes for the AI to analyze.

Together, these perspectives ensure that the AI algorithms have access to a richer array of visual information, improving their capacity to detect and differentiate between individual animals, even in a densely populated and dynamic environment such as a farm. This dual-view methodology also significantly widens the scope of detection, reducing blind spots and ensuring that the cows can be monitored effectively regardless of their orientation or position within the pen. The result is a robust dataset that simulates the multifaceted visual inputs required for a high-performing, real-world cattle monitoring system.

\subsubsection*{Data Annotation and Formation}

- Talk about how the data was collected using pipes and Amazon Ring Cameras

- How the data is annotated on Roboflow
The methodological rigor involved in data preparation is vital for the integrity of our machine learning model's training process. Here is a clear outline of the steps executed in preparing the dataset for cow detection in our investigation:

\begin{enumerate}
    \item \textbf{Frame Extraction:} We utilized a customized Python script for the extraction of frames from the video streams, converting them into a series of static images. To guarantee uniformity across the dataset, we extracted frames at regular one-second intervals. This process yielded `n' distinct images, which were then allocated to training, validation, and testing datasets for the subsequent stages of our machine learning endeavor.

    \item \textbf{Annotation with Roboflow:} We uploaded the frames onto Roboflow, a versatile annotation platform. This tool enabled our team to annotate images by meticulously outlining cows with bounding boxes, ensuring that the AI model can learn to identify the target objects effectively. The annotated frames are exemplified in Figure ~\ref{fig:camera-roboflow}.

    \item \textbf{Annotation Format Selection:} Roboflow's robust export options allowed us to obtain annotations in various formats suitable for different model architectures. We primarily opted for COCO and YOLOv5 formats, both widely recognized for their compatibility with state-of-the-art object detection algorithms.

    \item \textbf{Data Storage and Maintenance:} Post-annotation, we stored the images in the universally accepted JPG format. Accompanying these images, the corresponding annotation files were meticulously cataloged, readying the dataset for the intricate process of model training and subsequent evaluation.
\end{enumerate}
By adhering to these steps, we ensured the creation of a high-quality, standardized dataset poised for deployment in the development of an AI-powered cow detection system, geared towards enhancing the precision and efficiency of livestock management.

\subsection*{Simulation Design}

\subsubsection*{Data Splits}
We aim to thoroughly investigate model generalization across diverse conditions within livestock environments, specifically focusing on cattle localization. To achieve this, we meticulously designed and organized our dataset into five distinct configurations, each representing unique conditions under which the cattle were captured. These configurations are critical for evaluating the robustness and adaptability of object detection models, particularly in terms of their ability to generalize from one set of conditions to another. Below, we detail the dataset configurations and the rationale behind our data split strategy.

Dataset Configurations:
\begin{enumerate}
 \item\textbf{Top View:} Images captured from an overhead perspective, providing a comprehensive view of the livestock area.
 \item\textbf{Side View:} Images taken at a 60-degree (approx.) angle to the ground, offering a profile perspective of the cattle.
 \item\textbf{Daylight:} Images captured during daylight conditions from both the top and side views, ensuring natural lighting.
 \item\textbf{Nighttime:} Images obtained during nighttime from both the top and side views, with lighting conditions significantly reduced.
 \item\textbf{Breed Specific:} A subset of images exclusively featuring the Holstein breed, allowing for breed-specific model training.
 \end{enumerate}
Training and Testing Strategy
To rigorously assess model generalization, we employed a cross-testing methodology where models were trained on one dataset configuration and tested on another. This approach enabled us to isolate and understand the impact of various factors—such as viewing angle, lighting conditions, and breed variation—on model performance. The specific training and testing scenarios were as follows:
\begin{enumerate}
 \item\textbf{Viewing Angle Generalization:} Models were trained on the Top View dataset and tested on the Side View dataset, and vice versa. This setup assesses the model's ability to adapt to changes in perspective.
 \item\textbf{Lighting Condition Generalization:} Models trained on Daylight data were tested on Nighttime data to evaluate performance under varying lighting conditions, and vice versa.
\item\textbf{Breed Variation Generalization:} Models trained on the Breed Specific (Holstein) dataset were tested on a mixed-breed dataset (Holstein and Jersey), assessing the impact of breed diversity on detection accuracy.
 \item\textbf{Comprehensive Generalization:} Finally, models were trained on a combination of all dataset configurations to examine overall generalization capabilities across viewing angles, lighting conditions, and breed variations.
 \end{enumerate}
This structured approach to data split and testing is designed to provide insights into the extent to which object detection models, trained under specific conditions, can accurately generalize to different, untrained conditions. By systematically varying training and testing datasets, we aim to uncover potential limitations and strengths of current object detection technologies in the context of livestock monitoring, contributing valuable knowledge towards the development of more robust and adaptable solutions in precision agriculture.

\subsubsection*{Objective 1: How model performance is decomposed by different factors}
To investigate the model generalization. Factors such as lightning..

Each data configuration

\subsubsection*{Objective 2: How a fine-tuned model performed on a new dataset}

\subsection*{Model Training and Evaluation}

\subsubsection*{Trianing hyperparameters}

- how to cross validation is Design with different sample Size
- Iteration
- Evaluation metrics

\subsubsection*{Data Augmentation}

\subsubsection*{Model evaluation and cross validation}

% -------------------