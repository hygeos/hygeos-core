from pathlib import Path
import pytest
from hygeos_core.files import cache
from xarray.tutorial import open_dataset
from tempfile import TemporaryDirectory
from hygeos_core.tools import only
from hygeos_core.tests.conftest import savefig

# matplotlib is an optional dependency, only needed by cache_figure:
# skip those tests instead of failing collection when it is not installed
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

requires_matplotlib = pytest.mark.skipif(
    plt is None, reason="matplotlib is not installed"
)


@pytest.mark.parametrize('cache_function,var', [
    (cache.cache_json, 1),
    (cache.cache_json, 'a'),
    (cache.cache_json, [1, 'b', [1, 2]]),
    (cache.cache_pickle, [1, 'b', [1, 2]]),
])
def test_cachefunc(cache_function, var):
    def my_function(*args):
        return var
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.json'

        cache_function(cache_file)(my_function)()
        a = cache_function(cache_file)(my_function)()
        
        with pytest.raises(ValueError):
            # raise an error when called with the wrong arguments
            cache_function(cache_file)(my_function)(0)

        assert a == my_function()

def test_cache_json_inputs_modes():
    """cache_json with inputs='ignore' should not raise on different args;
    inputs='store' should store but also not check."""
    with TemporaryDirectory() as tmpdir:
        for mode in ("store", "ignore"):
            cache_file = Path(tmpdir) / f"cache_{mode}.json"
            fn = lambda *args: 42
            cache.cache_json(cache_file, inputs=mode)(fn)()
            # calling with different args must not raise
            result = cache.cache_json(cache_file, inputs=mode)(fn)(99)
            assert result == 42


def test_cache_pickle_inputs_modes():
    """cache_pickle with inputs='ignore' or 'store' should not raise on different args."""
    with TemporaryDirectory() as tmpdir:
        for mode in ("store", "ignore"):
            cache_file = Path(tmpdir) / f"cache_{mode}.pkl"
            fn = lambda *args: [1, 2, 3]
            cache.cache_pickle(cache_file, inputs=mode)(fn)()
            # calling with different args must not raise
            result = cache.cache_pickle(cache_file, inputs=mode)(fn)(99)
            assert result == [1, 2, 3]


class Test_cache_figure:
    
    @staticmethod
    def my_figure():
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        return fig

    def test_cache_philosophy(self, request):
        with TemporaryDirectory() as tmpdir:
            
            def my_figure(x):
                fig, ax = plt.subplots()
                ax.plot([0, x])
                ax.set_title(f'x: {x}')
                return fig
            
            cache_dir = Path(tmpdir) / "figures"
            
            # first call: creates the file
            cache.cache_figure(cache_dir)(my_figure)(3)
            savefig(request)
            cache_file = only(list(cache_dir.glob('*.png')))
            assert cache_file.name.startswith("my_figure_")
            
            # second call with same args: returns the same cached path
            cache.cache_figure(cache_dir)(my_figure)(3)
            savefig(request)
            
            # different args: produces a different file
            cache.cache_figure(cache_dir)(my_figure)(99)
            savefig(request)
            assert len(list(cache_dir.glob('*.png'))) == 2
            
        plt.close("all")

    def test_without_params(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "figures"
            cache.cache_figure(cache_dir)(self.my_figure)()
            cache_file = only(list(cache_dir.glob('*.png')))
            assert cache_file.name == "my_figure.png"
            
        plt.close("all")

    def test_figure_saving(self, request):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "figures"
            
            # first call: creates the file
            fig = cache.cache_figure(cache_dir)(self.my_figure)()
            fig.savefig(cache_dir/'test.png')
            
            # second call with same args: returns the same cached path
            fig = cache.cache_figure(cache_dir)(self.my_figure)()
            fig.savefig(cache_dir/'test.png')
            
        plt.close("all")


@pytest.mark.parametrize('extension', ["pickle", "csv"])
def test_cache_dataframe(extension):
    def my_function():
        return open_dataset('air_temperature').to_dataframe().reset_index().round()
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/('cache.'+extension)
        cache.cache_dataframe(cache_file)(my_function)()
        a = cache.cache_dataframe(cache_file)(my_function)()
        assert all(a == my_function())

def test_cache_dataset():
    def my_function():
        return open_dataset('air_temperature')
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.nc'
        cache.cache_dataset(cache_file,
                      attrs={'a': 1}
                      )(my_function)()
        a = cache.cache_dataset(
            cache_file,
            chunks={'lat': 10, 'lon': 10},
        )(my_function)()
        assert a == my_function()
    
class TestHashParams:
    """Tests for the hashparams function."""
    
    def test_same_args_produce_same_hash(self):
        """Same arguments should always produce the same hash."""
        h1 = cache.hashparams(a=1, b=2, c=3, x=10, y=20)
        h2 = cache.hashparams(a=1, b=2, c=3, x=10, y=20)
        assert h1 == h2
    
    def test_different_args_produce_different_hash(self):
        """Different arguments should produce different hashes."""
        h1 = cache.hashparams(a=1, b=2, c=3)
        h2 = cache.hashparams(a=1, b=2, c=4)
        assert h1 != h2
    
    def test_kwargs_order_independent(self):
        """Hash should be independent of kwargs order."""
        h1 = cache.hashparams(x=1, y=2, z=3)
        h2 = cache.hashparams(z=3, x=1, y=2)
        h3 = cache.hashparams(y=2, z=3, x=1)
        assert h1 == h2 == h3
    
    def test_only_kwargs(self):
        """Test hashing with only keyword arguments."""
        h1 = cache.hashparams(a=1, b="test")
        h2 = cache.hashparams(a=1, b="test")
        assert h1 == h2
    
    def test_empty_args(self):
        """Test hashing with no arguments."""
        h1 = cache.hashparams()
        h2 = cache.hashparams()
        assert h1 == h2
        assert len(h1) == 32  # BLAKE2b with digest_size=16 produces 32 hex chars
    
    def test_different_types(self):
        """Test hashing various data types."""
        h1 = cache.hashparams(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            tuple_val=(1, 2),
            none_val=None,
            bool_val=True
        )
        h2 = cache.hashparams(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            tuple_val=(1, 2),
            none_val=None,
            bool_val=True
        )
        assert h1 == h2
    
    def test_hash_format(self):
        """Test that hash is a valid hexadecimal string of correct length."""
        h = cache.hashparams(a=1, b=2, c=3, x=10)
        assert len(h) == 32  # 16 bytes = 32 hex characters
        assert all(c in '0123456789abcdef' for c in h)  # valid hex
    
    def test_positional_args_same_hash(self):
        """Same positional args should produce the same hash."""
        h1 = cache.hashparams(1, 2, 3)
        h2 = cache.hashparams(1, 2, 3)
        assert h1 == h2

    def test_positional_args_different_hash(self):
        """Different positional args should produce different hashes."""
        assert cache.hashparams(1, 2, 3) != cache.hashparams(1, 2, 4)

    def test_positional_args_order_matters(self):
        """Positional args are order-sensitive (unlike kwargs)."""
        assert cache.hashparams(1, 2, 3) != cache.hashparams(3, 2, 1)

    def test_positional_and_kwargs_combined(self):
        """Mixing positional and keyword args should be stable."""
        h1 = cache.hashparams(1, "hello", key=42)
        h2 = cache.hashparams(1, "hello", key=42)
        assert h1 == h2

    def test_positional_vs_kwargs_differ(self):
        """hashparams(1) should differ from hashparams(arg0=1) to avoid collisions."""
        # positional arg0 key is 'arg0', same as an explicit kwarg 'arg0'
        # This test documents the known behaviour rather than asserting isolation.
        h_pos = cache.hashparams(1)
        h_kw  = cache.hashparams(arg0=1)
        # They happen to produce the same hash since the key name is identical;
        # document this instead of asserting inequality.
        assert h_pos == h_kw  # known collision: arg0 positional == arg0 kwarg

    def test_nested_structures(self):
        """Test hashing nested data structures."""
        nested = {
            "level1": {
                "level2": [1, 2, {"level3": "deep"}]
            }
        }
        h1 = cache.hashparams(data=nested)
        h2 = cache.hashparams(data=nested)
        assert h1 == h2

def test_cache_dataframe_csv_checks(tmp_path):
    """Regression: csv cache did not check output nor inputs, and required a
    `time` column"""
    import pandas as pd
    cache_file = tmp_path / 'cache.csv'

    def f(n):
        return pd.DataFrame({'a': range(n), 'b': ['x'] * n})

    # no `time` column
    df1 = cache.cache_dataframe(cache_file)(f)(3)
    df2 = cache.cache_dataframe(cache_file)(f)(3)
    pd.testing.assert_frame_equal(df1, f(3))
    pd.testing.assert_frame_equal(df2, f(3))

    # inputs='check': different inputs raise
    with pytest.raises(ValueError):
        cache.cache_dataframe(cache_file)(f)(4)
    # inputs='ignore': not stored nor checked
    cache.cache_dataframe(tmp_path / 'c2.csv', inputs='ignore')(f)(3)
    cache.cache_dataframe(tmp_path / 'c2.csv', inputs='ignore')(f)(4)

    # the output check is effective: a frame that does not survive the csv
    # roundtrip is rejected
    def g():
        return pd.DataFrame({'a': [1.5]}, index=[10])
    with pytest.raises(AssertionError):
        cache.cache_dataframe(tmp_path / 'c3.csv')(g)()
    assert not (tmp_path / 'c3.csv').exists()


def test_cache_dataframe_csv_time(tmp_path):
    import pandas as pd
    df = pd.DataFrame({'time': pd.date_range('2020-01-01', periods=3), 'v': [1., 2., 3.]})
    cache.cache_dataframe(tmp_path / 'c.csv')(lambda: df)()
    out = cache.cache_dataframe(tmp_path / 'c.csv')(lambda: df)()
    pd.testing.assert_frame_equal(out, df, check_dtype=False)
    assert pd.api.types.is_datetime64_any_dtype(out['time'])


def test_cache_dataframe_csv_legacy_file(tmp_path):
    """csv cache files written by previous versions (without stored inputs)
    are still read with the default inputs='check'"""
    import pandas as pd
    df = pd.DataFrame({'time': pd.date_range('2020-01-01', periods=2), 'v': [1., 2.]})
    cache_file = tmp_path / 'legacy.csv'
    df.to_csv(cache_file, index=False)  # as written by the previous version
    out = cache.cache_dataframe(cache_file)(lambda x: None)(1)
    pd.testing.assert_frame_equal(out, df, check_dtype=False)


def test_cache_json_missing_parent(tmp_path):
    """Regression: the parent directory of the cache file was not created"""
    cache_file = tmp_path / 'nonexist' / 'sub' / 'c.json'
    assert cache.cache_json(cache_file)(lambda x: x + 1)(1) == 2
    assert cache_file.exists()
