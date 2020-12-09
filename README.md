<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Requirements](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project
Perform inference using an object detection model and TensorFlow

### Requirements
* Docker

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* Install Docker

### Installation
 
1. Clone the repo
```sh
git clone https://github.com/tachillon/Inference-object-detection-TensorFlow
```
2. Build the docker container
```sh
docker build -t <container_name>:<tag> .
```
<!-- USAGE EXAMPLES -->
## Usage
```sh
docker run --rm -it -v <path/to/workdir>:/tmp <container_name>:<tag> python3 /tmp/inference_on_images.py
```
```
inference_on_images.py/
├─ resultats/
├─ model/
│  ├─ frozen_inference_graph.pb
│  ├─ label.pbtxt
├─ images/
│  ├─ img1.jpg
│  ├─ img2.jpg
│  ├─ img3.jpg
```
Caution: the model to detect object is not provided. 

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Achille-Tâm GUILCHARD - achilletamguilchard@gmail.com
