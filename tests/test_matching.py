from agent.eval_tasks import any_form


def test_narrow_no_break_space_matches_plain_space():
    assert any_form(["Anaya Kapoor", "5"], "Anaya Kapoor has the most open tickets, with **5**.")


def test_thousands_separator_and_case():
    assert any_form(["123456"], "Revenue is 123,456 rupees")
    assert any_form(["Laptop Stand"], "the LAPTOP   stand")


def test_missing_value_fails():
    assert not any_form(["Kapoor", "7"], "Kapoor has 5 tickets")
