using System;
using System.Collections.Generic;

namespace TinyNN
{
    public static class Activations
    {
        public const string Sigmoid = nameof(Sigmoid);
        public const string Relu = nameof(Relu);
        public const string Tanh = nameof(Tanh);
    }

    public static class SigmoidActivation
    {
        public static double Calc(double val)
        {
            return 1.0 / (1.0 + Math.Exp(-val));
        }

        public static double CalcDer(double val)
        {
            return val * (1 - val);
        }
    }
    
    public static class TanhActivation
    {
        public static double Calc(double val)
        {
            return Math.Tanh(val);
        }

        public static double CalcDer(double val)
        {
            return 1 - (val * val);
        }
    }
    
    public static class ReluActivation
    {
        public static double Calc(double val)
        {
            return val > 0 ? val : 0;
        }

        public static double CalcDer(double val)
        {
            return val > 0 ? 1 : 0;
        }
    }
    
    public class Layer
    {
        public Layer(int neuronsCount, string activation)
        {
            NeuronsCount = neuronsCount;
            Activation = activation;
        }
        
        public int NeuronsCount;
        public string Activation;
    }
    
    public partial class Network
    {
        private const int biasVal = 1;

        private Layer[] _layers; //layers
        private readonly double _learningRate;

        private double[][][] _weights; //weights   
        private LCG _generator;

        public Network(Layer[] layers, double learningRate)
        {
            _generator = new LCG();
            _layers = layers;
            _learningRate = learningRate;

            InitWeights();
        }

        private void InitWeights()
        {
            List<double[][]> weightsList = new List<double[][]>();
            for (int layerIdx = 1; layerIdx < _layers.Length; layerIdx++)
            {
                List<double[]> layerWeightsList = new List<double[]>();
                int neuronsInPreviousLayer = _layers[layerIdx - 1].NeuronsCount;

                for (int layerNeuronIdx = 0; layerNeuronIdx < _layers[layerIdx].NeuronsCount; layerNeuronIdx++)
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
            var deltas = new double[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                deltas[i] = new double[_layers[i].NeuronsCount];
            }

            //calc deltas
            for (int layerIdx = _layers.Length - 1; layerIdx > 0; layerIdx--)
            {
                var layerNeuronsCount = _layers[layerIdx].NeuronsCount;
                var activation = _layers[layerIdx].Activation;
                
                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    if (layerIdx == _layers.Length - 1)
                    {
                        var val = nodeOutputs[layerIdx][layerNeuronIdx];
                        deltas[layerIdx][layerNeuronIdx] = ActivationDer(val, activation) * (val - outputs[layerNeuronIdx]);
                    }
                    else
                    {
                        var val = nodeOutputs[layerIdx][layerNeuronIdx];

                        double sum = 0;
                        int neuronsInNextLayer = _layers[layerIdx + 1].NeuronsCount;

                        for (int nextLayerNeuronIdx = 0; nextLayerNeuronIdx < neuronsInNextLayer; nextLayerNeuronIdx++)
                        {
                            sum += deltas[layerIdx + 1][nextLayerNeuronIdx] *
                                   _weights[layerIdx][nextLayerNeuronIdx][layerNeuronIdx];
                        }

                        deltas[layerIdx][layerNeuronIdx] = ActivationDer(val, activation) * sum;
                    }
                }
            }

            //update weights
            for (int layerIdx = _layers.Length - 1; layerIdx > 0; layerIdx--)
            {
                var layerNeuronsCount = _layers[layerIdx].NeuronsCount;

                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    int neuronsInPreviousLayer = _layers[layerIdx - 1].NeuronsCount;

                    for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < neuronsInPreviousLayer; prevLayerNeuronIdx++)
                    {
                        var dw = -_learningRate * deltas[layerIdx][layerNeuronIdx] *
                                 nodeOutputs[layerIdx - 1][prevLayerNeuronIdx];
                        _weights[layerIdx - 1][layerNeuronIdx][prevLayerNeuronIdx] += dw;
                    }

                    var biasDw = -_learningRate * deltas[layerIdx][layerNeuronIdx];
                    _weights[layerIdx - 1][layerNeuronIdx][neuronsInPreviousLayer] += biasDw;
                }
            }
        }

        private double[][] FeedForward(double[] inputs)
        {
            var nodeOutputs = new double[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                nodeOutputs[i] = new double[_layers[i].NeuronsCount];
            }

            for (var index = 0; index < inputs.Length; index++)
            {
                nodeOutputs[0][index] = inputs[index];
            }

            for (int layerIdx = 1; layerIdx < _layers.Length; layerIdx++)
            {
                var layerNeuronsCount = _layers[layerIdx].NeuronsCount;
                int neuronsInPreviousLayer = _layers[layerIdx - 1].NeuronsCount;
                
                var activationFn = _layers[layerIdx].Activation;

                for (int layerNeuronIdx = 0; layerNeuronIdx < layerNeuronsCount; layerNeuronIdx++)
                {
                    for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < neuronsInPreviousLayer; prevLayerNeuronIdx++)
                    {
                        var w = _weights[layerIdx - 1][layerNeuronIdx][prevLayerNeuronIdx];

                        nodeOutputs[layerIdx][layerNeuronIdx] += w * nodeOutputs[layerIdx - 1][prevLayerNeuronIdx];
                    }

                    var biasW = _weights[layerIdx - 1][layerNeuronIdx][neuronsInPreviousLayer]; //NOTE: bias
                    nodeOutputs[layerIdx][layerNeuronIdx] += biasW * biasVal;
                    
                    nodeOutputs[layerIdx][layerNeuronIdx] = Activation(nodeOutputs[layerIdx][layerNeuronIdx], activationFn);
                }
            }

            return nodeOutputs;
        }

        private double Activation(double val, string activation)
        {
            switch (activation)
            {
                case Activations.Sigmoid: return SigmoidActivation.Calc(val);
                case Activations.Relu: return ReluActivation.Calc(val);
                case Activations.Tanh: return TanhActivation.Calc(val);
            }
            
            throw new ArgumentException();
        }

        private double ActivationDer(double val, string activation)
        {
            switch (activation)
            {
                case Activations.Sigmoid: return SigmoidActivation.CalcDer(val);
                case Activations.Relu: return ReluActivation.CalcDer(val);
                case Activations.Tanh: return TanhActivation.CalcDer(val);
            }
            
            throw new ArgumentException();
        }
    }
}