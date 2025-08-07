"""
Test Visualization Agent Integration
"""

import pytest
from agents.visualization.agent import VisualizationAgent, VisualizationOptions

# Sample test data
TEST_DATA = [
    {"city": "NYC", "revenue": 1000},
    {"city": "London", "revenue": 1500},
    {"city": "Tokyo", "revenue": 800}
]

@pytest.fixture
def viz_agent():
    return VisualizationAgent()

def test_bar_chart_generation(viz_agent):
    """Test bar chart creation"""
    fig = viz_agent.visualize(
        data=TEST_DATA,
        options=VisualizationOptions(
            chart_type="bar",
            title="Test Revenue",
            x_axis="city",
            y_axis="revenue"
        )
    )
    assert fig is not None
    assert len(fig.data) == 1  # Should have one trace
    assert fig.layout.title.text == "Test Revenue"

def test_table_generation(viz_agent):
    """Test table creation"""
    fig = viz_agent.visualize(
        data=TEST_DATA,
        options=VisualizationOptions(chart_type="table")
    )
    assert fig is not None
    assert fig.data[0].type == "table"
    assert len(fig.data[0].header.values) == 2  # city + revenue

def test_export_function(viz_agent, tmp_path):
    """Test image export"""
    fig = viz_agent.visualize(
        data=TEST_DATA,
        options=VisualizationOptions(chart_type="bar")
    )
    export_path = tmp_path / "test_chart.png"
    viz_agent.export(fig, filename=str(export_path))
    assert export_path.exists()

# Integration test with ReasoningAgent
def test_reasoning_agent_integration():
    """Test full workflow with visualization"""
    from agents.reasoning_agent import ReasoningAgent
    from agents.query_executor.agent import QueryExecutorAgent
    
    # Mock database setup
    executor = QueryExecutorAgent("data/db.sqlite")
    reasoning_agent = ReasoningAgent(executor)
    
    # Test with visualization
    result = reasoning_agent.execute_query(
        sql="SELECT city, amount FROM bookings LIMIT 3",
        viz_options={
            "chart_type": "bar",
            "title": "Test Integration",
            "x_axis": "city",
            "y_axis": "amount"
        }
    )
    
    assert "visualization" in result
    assert result["visualization"].layout.title.text == "Test Integration" 