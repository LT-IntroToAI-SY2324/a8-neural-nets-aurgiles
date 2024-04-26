from neural import NeuralNet

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

voter_data = [
    ([.9, .6, .8, .3, .1], [1.0]),
    ([.8, .8, .4, .6, .4], [1.0]),
    ([.7, .2, .4, .6, .3], [1.0]),
    ([.5, .5, .8, .4, .8], [0.0]),
    ([.3, .1, .6, .8, .8], [0.0]),
    ([.6, .3, .4, .3, .6], [0.0])
]


voter_nn = NeuralNet(5, 10, 1)
voter_nn.train(voter_data)
print(voter_nn.test_with_expected(voter_data))

voter_test = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]

print(voter_nn.test(voter_test))