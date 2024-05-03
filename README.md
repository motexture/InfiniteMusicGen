# Continuous Music Generation

This Python script offers a seamless and non-interruptible music generation experience using audiocraft musicgen models. The music generation utilizes the last seconds of the previously generated song to maintain a continuous flow.

## Hardware requirements

- NVIDIA RTX 4090 GPU or higher for melody model

## Installation

1. **Clone the audiocraft repo locally**: git clone https://github.com/facebookresearch/audiocraft
3. **Navigate to the folder**: cd audiocraft
4. **Create a venv environment**: python -m venv myenv
5. **Activate your env**: source env/bin/activate # On Windows use env\Scripts\activate
6. **Install requirements**: pip install -r requirements.txt
7. **Install the latest torch cuda**: Follow installation steps from here https://pytorch.org/get-started/locally/
8. **Download the infigen.py script to your audiocraft folder**: Ensure to place it on the main folder
9. **Run the script**: python infigen.py
10. **Modify music genre**: Edit line 19
