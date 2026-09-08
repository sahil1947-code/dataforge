"""
tests/unit/test_calculator.py
------------------------------
Unit tests for the safe arithmetic calculator tool.
"""

import pytest
from tools.calculator import CalculatorTool


@pytest.mark.asyncio
async def test_basic_multiplication():
    t = CalculatorTool()
    r = await t.execute({"expression": "42 * 12"}, generation=1)
    assert r.success is True
    assert r.output["result"] == 504


@pytest.mark.asyncio
async def test_basic_addition():
    t = CalculatorTool()
    r = await t.execute({"expression": "5 + 5"}, generation=1)
    assert r.success is True
    assert r.output["result"] == 10


@pytest.mark.asyncio
async def test_division():
    t = CalculatorTool()
    r = await t.execute({"expression": "100 / 4"}, generation=1)
    assert r.success is True
    assert r.output["result"] == 25


@pytest.mark.asyncio
async def test_sqrt():
    t = CalculatorTool()
    r = await t.execute({"expression": "sqrt(144)"}, generation=1)
    assert r.success is True
    assert r.output["result"] == 12


@pytest.mark.asyncio
async def test_power():
    t = CalculatorTool()
    r = await t.execute({"expression": "2 ** 10"}, generation=1)
    assert r.success is True
    assert r.output["result"] == 1024


@pytest.mark.asyncio
async def test_empty_expression_fails():
    t = CalculatorTool()
    r = await t.execute({"expression": ""}, generation=1)
    assert r.success is False
    assert r.error is not None


@pytest.mark.asyncio
async def test_invalid_expression_fails():
    t = CalculatorTool()
    r = await t.execute({"expression": "import os"}, generation=1)
    assert r.success is False


@pytest.mark.asyncio
async def test_function_call_not_allowed():
    t = CalculatorTool()
    r = await t.execute({"expression": "open('/etc/passwd')"}, generation=1)
    assert r.success is False


@pytest.mark.asyncio
async def test_float_result():
    t = CalculatorTool()
    r = await t.execute({"expression": "1 / 3"}, generation=1)
    assert r.success is True
    assert abs(r.output["result"] - 0.3333333333) < 1e-6


@pytest.mark.asyncio
async def test_pi_constant():
    t = CalculatorTool()
    r = await t.execute({"expression": "pi * 2"}, generation=1)
    assert r.success is True
    assert abs(r.output["result"] - 6.283185307) < 1e-4
