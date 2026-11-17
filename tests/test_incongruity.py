from src.incongruity import features, feature_matrix

def test_sarcastic_example_flags_incongruity():
    f = features("Wah re sarkar, khaad toh aadhi mitti nikli!")
    assert f[2] == 1.0

def test_plain_question_does_not():
    assert features("Which fertilizer is best for wheat in sandy soil?")[2] == 0.0

def test_matrix_shape():
    assert feature_matrix(["a", "b", "c"]).shape == (3, 4)
