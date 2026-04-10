"""


So how is the model divided between thousands of GPU is there a standardized rule which has to be followed 



"How will the system know which GPU has which transformer layer 



Also how does a system know which gpu to send data to 





According to my understanding 



user query -> input data to first GPU - > get response calculated weights from GPU1 back to system -> weights then send to GPU 2 



does it work this way or is the process automated where system input the query tensors to GPU1 and then it automatically passed through the network of GPU and then result is outputted 



"""