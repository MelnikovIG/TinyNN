using System;
using System.Linq;

namespace TinyNN.Example.Sum
{
    public class Program
    {
        public static void Main()
        {
            Layer[] layers = {
                new Layer(InputsLength, Activations.Sigmoid),
                new Layer(10, Activations.Sigmoid),
                new Layer(5, Activations.Sigmoid),
                new Layer(1, Activations.Relu),
            };
            
            var network = new Network(layers, 0.1);
            
            //Train
            for (int i = 0; i < 1_000_000; i++)
            {
                var data = GetAvgData();

                var error = network.Train(data.input, data.output);

                if (i % 100_000 == 0)
                {
                    Console.WriteLine($"Err {error}");
                }
            }
            
            Console.WriteLine($"\r\n Start test \r\n");
            
            //Test
            for (int i = 0; i < 10; i++)
            {
                var data = GetAvgData();

                var predicted = network.Predict(data.input)[0];
                var actual = data.output[0];
                
                Console.Error.WriteLine($"P {predicted} A {actual} DIFF {predicted - actual}");
            }
        }

        private const int InputsLength = 5;
        static Random rnd = new Random();
        private static (double[] input, double[] output) GetAvgData()
        {
            var inp = new double[InputsLength];
            for (int i = 0; i < InputsLength; i++)
            {
                inp[i] = rnd.NextDouble();
            }

            var @out = inp.Sum();

            return (inp, new[] {@out});
        }
    }
}