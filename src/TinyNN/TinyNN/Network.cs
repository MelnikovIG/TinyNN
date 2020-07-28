using System;
using System.Collections.Generic;

namespace TinyNN
{
    public class Network
    {
        private const int biasVal = 1;

        private int[] _layers; //layers

        private double[][][] _weights; //weights   
        private LCG _generator;

        public Network(NetworkSettings settings)
        {
            _generator = new LCG();

            InitLayers(settings);
            InitWeights();
        }

        private void InitWeights()
        {
            List<double[][]> weightsList = new List<double[][]>();
            for (int layerIdx = 1; layerIdx < _layers.Length; layerIdx++)
            {
                List<double[]> layerWeightsList = new List<double[]>();
                int neuronsInPreviousLayer = _layers[layerIdx - 1];

                for (int layerNeuronIdx = 0; layerNeuronIdx < _layers[layerIdx]; layerNeuronIdx++)
                {
                    double[] neuronWeights = new double[neuronsInPreviousLayer + 1];
                    for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < neuronsInPreviousLayer; prevLayerNeuronIdx++)
                    {
                        neuronWeights[prevLayerNeuronIdx] = _generator.NextNormalized();
                    }

                    neuronWeights[neuronsInPreviousLayer] = _generator.NextNormalized(); //NOTE: bias

                    layerWeightsList.Add(neuronWeights);
                }

                weightsList.Add(layerWeightsList.ToArray());
            }

            _weights = weightsList.ToArray();
        }

        private void InitLayers(NetworkSettings settings)
        {
            var layers = new List<int>();
            layers.Add(settings.InputsCount);
            foreach (var hiddenLayer in settings.HiddenLayers)
            {
                layers.Add(hiddenLayer);
            }

            layers.Add(settings.OutputsCount);

            _layers = layers.ToArray();
        }


        public double[] Predict(double[] inputs)
        {
            var nodeOutputs = FeedForward(inputs);
            return nodeOutputs[_layers.Length - 1];
        }

        public double Train(double[] inputs, double[] outputs)
        {
            var nodeOutputs = FeedForward(inputs);
            
            BackPropagate(outputs, nodeOutputs);

            var error = CalculateError(nodeOutputs[_layers.Length - 1], outputs);
            return error;
        }

        private double CalculateError(double[] lastLayerOutputs, double[] outputs)
        {
            var sumSqrErr = 0.0;
            for (int i = 0; i < outputs.Length; i++)
            {
                sumSqrErr += Math.Pow(outputs[i] - lastLayerOutputs[i], 2);
            }
            var error = 0.5 * sumSqrErr;
            return error;
        }
        
        private void BackPropagate(double[] outputs, double[][] nodeOutputs)
        {
            const double learningRate = 0.5;

            var deltas = new double[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                deltas[i] = new double[_layers[i]];
            }

            //calc deltas
            for (int layerIdx = _layers.Length - 1; layerIdx > 0; layerIdx--)
            {
                var layerNeuronsCount = _layers[layerIdx];

                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    if (layerIdx == _layers.Length - 1)
                    {
                        var val = nodeOutputs[layerIdx][layerNeuronIdx];
                        deltas[layerIdx][layerNeuronIdx] = val * (1 - val) * (val - outputs[layerNeuronIdx]);
                    }
                    else
                    {
                        var val = nodeOutputs[layerIdx][layerNeuronIdx];

                        double sum = 0;
                        int neuronsInNextLayer = _layers[layerIdx + 1];

                        for (int nextLayerNeuronIdx = 0; nextLayerNeuronIdx < neuronsInNextLayer; nextLayerNeuronIdx++)
                        {
                            sum += deltas[layerIdx + 1][nextLayerNeuronIdx] *
                                   _weights[layerIdx][nextLayerNeuronIdx][layerNeuronIdx];
                        }

                        deltas[layerIdx][layerNeuronIdx] = val * (1 - val) * sum;
                    }
                }
            }

            //update weights
            for (int layerIdx = _layers.Length - 1; layerIdx > 0; layerIdx--)
            {
                var layerNeuronsCount = _layers[layerIdx];

                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    int neuronsInPreviousLayer = _layers[layerIdx - 1];

                    for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < neuronsInPreviousLayer; prevLayerNeuronIdx++)
                    {
                        var dw = -learningRate * deltas[layerIdx][layerNeuronIdx] *
                                 nodeOutputs[layerIdx - 1][prevLayerNeuronIdx];
                        _weights[layerIdx - 1][layerNeuronIdx][prevLayerNeuronIdx] += dw;
                    }

                    var biasDw = -learningRate * deltas[layerIdx][layerNeuronIdx];
                    _weights[layerIdx - 1][layerNeuronIdx][neuronsInPreviousLayer] += biasDw;
                }
            }
        }

        private double[][] FeedForward(double[] inputs)
        {
            var nodeOutputs = new double[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                nodeOutputs[i] = new double[_layers[i]];
            }

            for (var index = 0; index < inputs.Length; index++)
            {
                nodeOutputs[0][index] = inputs[index];
            }

            for (int layerIdx = 1; layerIdx < _layers.Length; layerIdx++)
            {
                var layerNeuronsCount = _layers[layerIdx];
                int neuronsInPreviousLayer = _layers[layerIdx - 1];

                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < neuronsInPreviousLayer; prevLayerNeuronIdx++)
                    {
                        var w = _weights[layerIdx - 1][layerNeuronIdx][prevLayerNeuronIdx];

                        nodeOutputs[layerIdx][layerNeuronIdx] += w * nodeOutputs[layerIdx - 1][prevLayerNeuronIdx];
                    }

                    var biasW = _weights[layerIdx - 1][layerNeuronIdx][neuronsInPreviousLayer]; //NOTE: bias
                    nodeOutputs[layerIdx][layerNeuronIdx] += biasW * biasVal;

                    nodeOutputs[layerIdx][layerNeuronIdx] = Sigmoid(nodeOutputs[layerIdx][layerNeuronIdx]);
                }
            }

            return nodeOutputs;
        }

        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
    }
}