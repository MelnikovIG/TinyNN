namespace TinyNN
{
    //linear congruential generator
    public class LCG
    {
        private const long m = 0x80000000; // 2^31 - 2 147 483 648
        private const long a = 1103515245;
        private const long c = 12345;
        private long _last;

        public LCG()
        {
            _last = 1103527590;
        }

        public long Next()
        {
            var cur = _last;

            _last = (a * _last + c) % m;

            return cur;
        }

        public long Next(long maxValue)
        {
            return Next() % maxValue;
        }

        public double NextNormalized() => Next() / (double) 0x7fffffff;
    }
}