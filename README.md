# Gender Prediction given names
Gender prediction from Indian names.
The trained model tries to predict the gender of the person given the name of the person. The model has been trained only with Indian names for now.

## Using the model:

You need python3.6 and tensorflow 1.8 for running the model.

clone the repo:

`git clone https://github.com/mandalbiswadip/genderPrediction.git`

`cd genderPrediction`

Create a virtual environment if you like to work in one and install the requirements:

`pip3 install -r requirements.txt`

Put the project path in `scripts.paths.py` in the `HOME` variable. It should look like this:

`HOME = "folder/genderPrediction"`

Finally, get the prediction with prediction probabilities:

```buildoutcfg
python3 classify.py animesh
>>Gender for animesh: [{'male': 0.92938721, 'female': 0.070612811}]

```

Model description:
Character level embedding(One hot embedding) has been used and has been fed to a sequential Recurrent Neural Network(bidirectional).
Currently, the model is trained for names with maximum 20 characters.



Sample Results:

For a given name, probability of being the gender 'male' or 'female' predicted by the model.

name: 'biswadip'
`
{
'male': 0.9898892, 
'female': 0.01011086
}
`

name: 'Aadarsh'
`
{
'male': 0.9724812, 
'female': 0.02751883
}`

name: 'Maadhav'
`
{
'male': 0.97186756, 
'female': 0.02813238
}`

name: 'Pabitra'
`
{
'male': 0.270486, 
'female': 0.729514
}`
