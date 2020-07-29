using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace TinyNN.Tests
{
    public class NetworkTests
    {
        private readonly ITestOutputHelper _output;

        public NetworkTests(ITestOutputHelper output)
        {
            _output = output;
        }
        
        [Fact]
        public void TestNoHiddenLayers()
        {
            var settings = ReadSettingsHardcoded();
            
            _output.WriteLine(settings);
            
            NetworkSettings networkSettings = GetNetworkSettings(settings);

            List<Layer> layers = new List<Layer>();
            layers.Add(new Layer(networkSettings.InputsCount, Activations.Sigmoid));
            foreach (var hiddenLayerNeurons in networkSettings.HiddenLayers)
            {
                layers.Add(new Layer(hiddenLayerNeurons, Activations.Sigmoid));
            }
            layers.Add(new Layer(networkSettings.OutputsCount, Activations.Sigmoid));

            Network network = new Network(layers.ToArray(), 0.5);
            
            for (int trainingIteration = 0; trainingIteration < networkSettings.TrainingIterations; trainingIteration++)
            {
                for (int i = 0; i < networkSettings.TrainingExamplesCount; i++)
                {
                    var err = network.Train(networkSettings.TrainingInputs[i].Data, networkSettings.ExpectedOutputs[i].Data);
                }
            }

            var expectedTestResults = new[]
            {
                0.4968185422996499,
                0.68336702766874
            };

            var idx = 0;
            foreach (var testInput in networkSettings.TestInputs)
            {
                var result = network.Predict(testInput.Data);

                var res = result[0];
                
                Assert.True(Math.Abs(expectedTestResults[idx] - res) < 0.00000000001);

                idx++;
            }
        }
        
        [Fact]
        public void TestWithHiddenLayers()
        {
            var settings = ReadSettingsHardcoded2();
            
            Console.Error.WriteLine(settings);
            
            NetworkSettings networkSettings = GetNetworkSettings(settings);
            List<Layer> layers = new List<Layer>();
            layers.Add(new Layer(networkSettings.InputsCount, Activations.Sigmoid));
            foreach (var hiddenLayerNeurons in networkSettings.HiddenLayers)
            {
                layers.Add(new Layer(hiddenLayerNeurons, Activations.Sigmoid));
            }
            layers.Add(new Layer(networkSettings.OutputsCount, Activations.Sigmoid));

            Network network = new Network(layers.ToArray(), 0.5);
            
            for (int trainingIteration = 0; trainingIteration < networkSettings.TrainingIterations; trainingIteration++)
            {
                for (int i = 0; i < networkSettings.TrainingExamplesCount; i++)
                {
                    var err = network.Train(networkSettings.TrainingInputs[i].Data, networkSettings.ExpectedOutputs[i].Data);
                }
            }

            var expectedTestResults = new[]
            {
                0.02390042747686394,
                0.11046779833635939,
                0.7370635517892113,
                0.9637416082116796
            };

            var idx = 0;
            foreach (var testInput in networkSettings.TestInputs)
            {
                var result = network.Predict(testInput.Data);

                var res = result[0];
                
                Assert.True(Math.Abs(expectedTestResults[idx] - res) < 0.00000000001);

                idx++;
            }
        }
 
        private static NetworkSettings GetNetworkSettings(string settingsStr)
        {
            var settings = new NetworkSettings();
            
            var lineIdx = 0;
            var lines = settingsStr.Split(new[] {Environment.NewLine}, StringSplitOptions.None);
            
            var inputs = lines[lineIdx++].Split(' ');
            settings.InputsCount = int.Parse(inputs[0]);
            settings.OutputsCount = int.Parse(inputs[1]);
            settings.HiddenLayersCount = int.Parse(inputs[2]);
            settings.TestInputsCount = int.Parse(inputs[3]);
            settings.TrainingExamplesCount = int.Parse(inputs[4]);
            settings.TrainingIterations = int.Parse(inputs[5]);
            inputs = lines[lineIdx++].Split(' ');
            
            settings.HiddenLayers = new int[settings.HiddenLayersCount];
            for (int i = 0; i < settings.HiddenLayersCount; i++)
            {
                settings.HiddenLayers[i] = int.Parse(inputs[i]);
            }
            
            settings.TestInputs = new InputData[settings.TestInputsCount];
            for (int i = 0; i < settings.TestInputsCount; i++)
            {
                settings.TestInputs[i] = new InputData(lines[lineIdx++].Select(x => (double)CharUnicodeInfo.GetDigitValue(x)).ToArray());
            }
            
            settings.TrainingInputs = new InputData[settings.TrainingExamplesCount];
            settings.ExpectedOutputs = new OutputData[settings.TrainingExamplesCount];
            for (int i = 0; i < settings.TrainingExamplesCount; i++)
            {
                inputs = lines[lineIdx++].Split(' ');
                settings.TrainingInputs[i] = new InputData(inputs[0].Select(x => (double)CharUnicodeInfo.GetDigitValue(x)).ToArray());
                settings.ExpectedOutputs[i] = new OutputData(inputs[1].Select(x => (double)CharUnicodeInfo.GetDigitValue(x)).ToArray());
            }
            return settings;
        }
        
        private static string ReadSettingsHardcoded()
        {
            return @"1 1 0 2 2 7

0
1
0 0
1 1";
        }
        
        private static string ReadSettingsHardcoded2()
        {
            return @"16 1 1 4 80 100
4
1000001010101010
0000111000101101
1101000001101011
1100111000110010
0011001001110000 0
0100000111011111 1
1000010111001100 0
1001010110011010 1
0011100011110011 1
1010110001110100 1
0110000011010100 0
0010001111010101 1
0001111010001100 0
1101001000100001 0
0010111101001000 0
0010011110110101 1
0001010110110110 1
0110010001011010 0
1110010111011010 1
0110010011000011 0
0011110110111000 1
1001000011001010 0
0111101001101110 1
1110011110011110 1
1001011011101011 1
0010011001000000 0
0101101000010000 0
0100010010101101 0
0111001010011000 0
1100111000001010 0
1101100111001010 1
1010111111000111 1
1100000100101101 0
0111010000000001 0
1000001001110001 0
1010110011111110 1
1100001101110000 0
1100010001000000 0
1001100111010111 1
0110001000100101 0
1110011101001000 1
1110110100011100 1
1000111111100001 1
0100000000011011 0
1000110001010000 0
0011101000001001 0
0110101111011011 1
1011000011101101 1
0011000000000010 0
0100100101101010 0
0010000011100011 0
0011011001100111 1
0000101101101100 0
0111111110011110 1
1110011010000100 0
0101001000100010 0
1000010011100101 0
0101110111011100 1
1010000001100101 0
0011011010110101 1
0101110110100010 1
1011111011101010 1
1100001111010101 1
1000111110111111 1
0010010110101101 1
1110011111000010 1
0111011111011110 1
0011011000000011 0
0000011001100011 0
0101001000011011 0
0100101100101101 1
1101110001110011 1
0011010100110101 1
0001101101110011 1
0110011000011011 1
0100010010100110 0
1000101110000110 0
1101100010000101 0
1010010101100111 1
0101001100110100 0
1001010111101111 1
1111101101100111 1
0100111101101010 1
0001000101001011 0";
        }
    }
}