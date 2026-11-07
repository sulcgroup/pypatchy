"""
Fast, offline unit tests for the pure-logic parameter-value machinery used throughout
PatchySimulationEnsemble: ParameterValue/ParamValueGroup (base_param_val.py), the
parameter_value() factory and EnsembleParameter/EnsembleParameters (ensemble_parameter.py),
and ParamSet/PatchySimulation (simulation_specification.py).

None of these tests touch the filesystem, oxDNA, or a server config - they should run in
well under a second.
"""
import pytest

from pypatchy.patchy.base_param_val import (
    ParameterValue,
    ParamValueGroup,
    InvalidParameterError,
    InvalidParameterTypeError,
)
from pypatchy.patchy.ensemble_parameter import (
    parameter_value,
    EnsembleParameter,
    EnsembleParameters,
)
from pypatchy.patchy.simulation_specification import ParamSet, PatchySimulation, NoSuchParamError


# --------------------------- ParameterValue --------------------------- #

def test_parameter_value_equality_same_name_and_value():
    assert ParameterValue("T", 0.1) == ParameterValue("T", 0.1)


def test_parameter_value_equality_different_name():
    assert ParameterValue("T", 0.1) != ParameterValue("density", 0.1)


def test_parameter_value_equality_different_value():
    assert ParameterValue("T", 0.1) != ParameterValue("T", 0.2)


def test_parameter_value_equality_against_raw_value():
    # __eq__ falls back to comparing param_value directly against non-ParameterValue others
    pv = ParameterValue("T", 0.1)
    assert pv == 0.1
    assert pv != 0.2
    assert pv != "not a number"


def test_parameter_value_hashable_and_usable_in_sets():
    a = ParameterValue("T", 0.1)
    b = ParameterValue("T", 0.1)
    c = ParameterValue("T", 0.2)
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


def test_parameter_value_str_and_str_verbose():
    pv = ParameterValue("T", 0.1)
    assert str(pv) == "T_0.1"
    assert pv.str_verbose() == "T: 0.1"


def test_parameter_value_lt_against_same_name_parameter_value():
    assert ParameterValue("T", 0.1) < ParameterValue("T", 0.2)
    assert not (ParameterValue("T", 0.2) < ParameterValue("T", 0.1))


def test_parameter_value_lt_against_mismatched_name_raises():
    with pytest.raises(ValueError):
        ParameterValue("T", 0.1) < ParameterValue("density", 0.2)


def test_parameter_value_lt_against_raw_value():
    assert ParameterValue("T", 0.1) < 0.2


def test_parameter_value_lt_against_incompatible_type_raises():
    with pytest.raises(TypeError):
        ParameterValue("T", 0.1) < object()


def test_parameter_value_rejects_nested_parameter_value():
    # __post_init__ guards against constructing a ParameterValue whose value is itself
    # a ParameterValue - this should raise, not silently nest
    with pytest.raises(InvalidParameterError):
        ParameterValue("T", ParameterValue("density", 0.1))


# --------------------------- Invalid*Error constructors --------------------------- #
# regression tests for constructor bugs found while adding this coverage:
# InvalidParameterError never stored `custom`, and InvalidParameterTypeError's __init__
# called InvalidParameterError.__init__ without `self`, corrupting the arguments.

def test_invalid_parameter_error_str_includes_custom_message():
    err = InvalidParameterError("T", "bogus", "extra detail")
    assert "extra detail" in str(err)
    assert "T" in str(err)


def test_invalid_parameter_error_str_without_custom_message():
    err = InvalidParameterError("T", "bogus")
    # should not raise, and should fall back to the generic message
    assert "T" in str(err)


def test_invalid_parameter_type_error_constructs_correctly():
    err = InvalidParameterTypeError("cvt_settings", 5, "must be a MultidentateConvertSettings")
    assert isinstance(err, TypeError)
    assert isinstance(err, InvalidParameterError)
    assert "must be a MultidentateConvertSettings" in str(err)


# --------------------------- ParamValueGroup --------------------------- #

def _make_group():
    return ParamValueGroup(
        param_name="staging",
        param_value={
            "t": ParameterValue("t", 0),
            "T": ParameterValue("T", 0.1),
        },
        valname="stage1",
    )


def test_param_value_group_getitem_and_has_param():
    grp = _make_group()
    assert grp["t"] == 0
    assert grp["T"] == 0.1
    assert grp.has_param("t")
    assert not grp.has_param("nonexistent")


def test_param_value_group_contains_str_and_parameter_value():
    grp = _make_group()
    assert "t" in grp
    assert "nonexistent" not in grp
    assert ParameterValue("t", 0) in grp


def test_param_value_group_value_name_is_valname():
    grp = _make_group()
    assert grp.value_name() == "stage1"


def test_param_value_group_iter_flattens_nested_groups():
    inner = _make_group()
    outer = ParamValueGroup(
        param_name="outer",
        param_value={"inner": inner, "x": ParameterValue("x", 1)},
        valname="outer1",
    )
    flattened = list(outer)
    assert ParameterValue("x", 1) in flattened
    assert ParameterValue("t", 0) in flattened
    assert ParameterValue("T", 0.1) in flattened
    assert inner not in flattened  # nested group itself is not yielded, its members are


def test_param_value_group_eq_against_raw_value_does_not_crash():
    # regression test: __eq__ used to assume `other` was always a ParameterValue and
    # crashed with AttributeError when compared against a plain value
    grp = _make_group()
    assert grp != "not a parameter value"
    assert grp != 5


def test_param_value_group_eq_against_matching_group():
    a = _make_group()
    b = _make_group()
    assert a == b


def test_param_value_group_eq_against_differing_group():
    a = _make_group()
    b = ParamValueGroup(
        param_name="staging",
        param_value={"t": ParameterValue("t", 1), "T": ParameterValue("T", 0.1)},
        valname="stage2",
    )
    assert a != b


# --------------------------- parameter_value() factory --------------------------- #

def test_parameter_value_factory_basic_scalar():
    pv = parameter_value("T", 0.1)
    assert isinstance(pv, ParameterValue)
    assert not isinstance(pv, ParamValueGroup)
    assert pv.param_name == "T"
    assert pv.param_value == 0.1


def test_parameter_value_factory_group_without_explicit_name():
    pv = parameter_value("stage1", {"t": 0, "T": 0.1})
    assert isinstance(pv, ParamValueGroup)
    assert pv.param_name == "stage1"
    assert pv.value_name() == "stage1"  # falls back to the dict key
    assert pv["t"] == 0
    assert pv["T"] == 0.1


def test_parameter_value_factory_group_with_explicit_name():
    pv = parameter_value("stage1", {"name": "custom_name", "t": 0})
    assert isinstance(pv, ParamValueGroup)
    assert pv.param_name == "stage1"  # dict key, unaffected by "name"
    assert pv.value_name() == "custom_name"
    assert "name" not in pv.group_params_names()  # "name" key itself is not a sub-param


def test_parameter_value_factory_invalid_object_type_raises():
    with pytest.raises(Exception):
        parameter_value("x", {"type": "not_a_real_type"})


def test_parameter_value_factory_staging_missing_stages_key_raises_with_useful_message():
    # regression test: this used to KeyError on `data['name']` instead of raising the
    # intended TypeError, because "name" is deliberately excluded from `data`
    with pytest.raises(TypeError) as excinfo:
        parameter_value("staging", {"type": "staging"})
    assert "staging" in str(excinfo.value)


# --------------------------- EnsembleParameter / EnsembleParameters --------------------------- #

def _make_ensemble_parameter() -> EnsembleParameter:
    return EnsembleParameter("T", [ParameterValue("T", 0.05), ParameterValue("T", 0.1)])


def test_ensemble_parameter_rejects_duplicate_values():
    with pytest.raises(AssertionError):
        EnsembleParameter("T", [ParameterValue("T", 0.1), ParameterValue("T", 0.1)])


def test_ensemble_parameter_lookup_by_string():
    ep = _make_ensemble_parameter()
    assert ep.lookup("0.05") == ParameterValue("T", 0.05)


def test_ensemble_parameter_lookup_missing_raises_keyerror():
    ep = _make_ensemble_parameter()
    with pytest.raises(KeyError):
        ep.lookup("999")


def test_ensemble_parameter_getitem_int_and_str():
    ep = _make_ensemble_parameter()
    assert ep[0] == ParameterValue("T", 0.05)
    assert ep["0.1"] == ParameterValue("T", 0.1)


def test_ensemble_parameter_contains():
    ep = _make_ensemble_parameter()
    assert ParameterValue("T", 0.05) in ep
    assert ParameterValue("T", 0.5) not in ep


def test_ensemble_parameter_len_and_iter():
    ep = _make_ensemble_parameter()
    assert len(ep) == 2
    assert list(ep) == [ParameterValue("T", 0.05), ParameterValue("T", 0.1)]


def test_ensemble_parameter_is_grouped_params():
    ep = _make_ensemble_parameter()
    assert not ep.is_grouped_params()


def test_ensemble_parameters_getitem_by_name_and_index():
    eps = EnsembleParameters([
        _make_ensemble_parameter(),
        EnsembleParameter("duplicate", [ParameterValue("duplicate", 0), ParameterValue("duplicate", 1)]),
    ])
    assert eps["T"].param_key == "T"
    assert eps[1].param_key == "duplicate"


def test_ensemble_parameters_rejects_duplicate_keys():
    with pytest.raises(AssertionError):
        EnsembleParameters([_make_ensemble_parameter(), _make_ensemble_parameter()])


def test_ensemble_parameters_contains():
    eps = EnsembleParameters([_make_ensemble_parameter()])
    assert "T" in eps
    assert "density" not in eps


def test_ensemble_parameters_set_param_adds_new():
    eps = EnsembleParameters([_make_ensemble_parameter()])
    new_param = EnsembleParameter("duplicate", [ParameterValue("duplicate", 0)])
    eps.set_param(new_param)
    assert len(eps) == 2
    assert eps["duplicate"] is new_param


def test_ensemble_parameters_set_param_replaces_existing():
    eps = EnsembleParameters([_make_ensemble_parameter()])
    replacement = EnsembleParameter("T", [ParameterValue("T", 999)])
    eps.set_param(replacement)
    assert len(eps) == 1
    assert eps["T"] is replacement


def test_ensemble_parameters_delitem():
    eps = EnsembleParameters([
        _make_ensemble_parameter(),
        EnsembleParameter("duplicate", [ParameterValue("duplicate", 0)]),
    ])
    del eps["T"]
    assert "T" not in eps
    assert len(eps) == 1
    assert eps["duplicate"].param_key == "duplicate"


def test_ensemble_parameters_delitem_missing_raises_keyerror():
    eps = EnsembleParameters([_make_ensemble_parameter()])
    with pytest.raises(KeyError):
        del eps["nonexistent"]


# --------------------------- ParamSet / PatchySimulation --------------------------- #

def test_paramset_contains_and_getitem():
    ps = ParamSet([ParameterValue("T", 0.1), ParameterValue("density", 0.05)])
    assert "T" in ps
    assert "nonexistent" not in ps
    assert ps["T"] == 0.1


def test_paramset_setitem_replaces_existing():
    ps = ParamSet([ParameterValue("T", 0.1)])
    ps["T"] = 0.2
    assert ps["T"] == 0.2
    assert len(ps.param_vals) == 1


def test_paramset_setitem_adds_new():
    ps = ParamSet([ParameterValue("T", 0.1)])
    ps["density"] = 0.05
    assert ps["density"] == 0.05
    assert len(ps.param_vals) == 2


def test_paramset_append_rejects_duplicate():
    ps = ParamSet([ParameterValue("T", 0.1)])
    with pytest.raises(AssertionError):
        ps.append(ParameterValue("T", 0.2))


def test_paramset_add_merges_non_overlapping():
    a = ParamSet([ParameterValue("T", 0.1)])
    b = ParamSet([ParameterValue("density", 0.05)])
    merged = a + b
    assert merged["T"] == 0.1
    assert merged["density"] == 0.05


def test_paramset_add_prefers_left_on_overlap():
    a = ParamSet([ParameterValue("T", 0.1)])
    b = ParamSet([ParameterValue("T", 0.9), ParameterValue("density", 0.05)])
    merged = a + b
    assert merged["T"] == 0.1  # left side wins, right side's overlapping key dropped
    assert merged["density"] == 0.05


def test_paramset_equivalent_order_independent():
    a = ParamSet([ParameterValue("T", 0.1), ParameterValue("density", 0.05)])
    b = ParamSet([ParameterValue("density", 0.05), ParameterValue("T", 0.1)])
    assert a.equivelant(b)


def test_paramset_equivalent_false_on_mismatch():
    a = ParamSet([ParameterValue("T", 0.1)])
    b = ParamSet([ParameterValue("T", 0.2)])
    assert not a.equivelant(b)


def test_paramset_eq_is_order_dependent_via_repr():
    a = ParamSet([ParameterValue("T", 0.1), ParameterValue("density", 0.05)])
    b = ParamSet([ParameterValue("density", 0.05), ParameterValue("T", 0.1)])
    # __eq__ compares repr(), which is order-dependent - unlike equivelant()
    assert a != b


def test_paramset_get_folder_path():
    ps = ParamSet([ParameterValue("T", 0.1), ParameterValue("duplicate", 0)])
    assert ps.get_folder_path().parts == ("T_0.1", "duplicate_0")


def test_patchysimulation_is_paramset_alias():
    assert PatchySimulation is ParamSet


def test_no_such_param_error_message_includes_param_name():
    class FakeEnsemble:
        def long_name(self):
            return "fake_ensemble"

    err = NoSuchParamError(FakeEnsemble(), "T")
    assert "T" in str(err)
    assert "fake_ensemble" in str(err)


def test_no_such_param_error_includes_sim_when_set():
    class FakeEnsemble:
        def long_name(self):
            return "fake_ensemble"

    err = NoSuchParamError(FakeEnsemble(), "T")
    sim = ParamSet([ParameterValue("duplicate", 0)])
    err.set_sim(sim)
    assert repr(sim) in str(err)