import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =========================================================
# INTERACTIVE POLYNOMIAL TUTOR — STREAMLIT
# Lessons + questions + hints + answer checking + score
# Real coefficients only; real and complex roots supported
# Maximum polynomial degree: 20
# =========================================================

st.set_page_config(
    page_title="Interactive Polynomial Tutor",
    page_icon="📘",
    layout="wide"
)

x = sp.symbols("x")
MAX_DEGREE = 20

NAVY = "#142D4E"
BLUE = "#1769AA"
GREEN = "#167D46"
RED = "#B42318"

st.markdown(f"""
<style>
.stApp {{ background-color: #F2F5F9; }}
.main-title {{
    background: {NAVY}; color: white; padding: 1.2rem;
    border-radius: 12px; text-align: center; margin-bottom: 1rem;
}}
.main-title h1 {{ margin: 0; font-size: 2rem; }}
.main-title p {{ color: #D6E5F5; margin: .4rem 0 0 0; }}
.lesson-card {{
    background: white; border-radius: 10px; padding: 1rem;
    border: 1px solid #D9E2EC; margin-bottom: .8rem;
}}
</style>
<div class="main-title">
  <h1>INTERACTIVE POLYNOMIAL TUTOR</h1>
  <p>Learn • Practise • Receive Feedback • Understand</p>
</div>
""", unsafe_allow_html=True)


def parse_coefficients(text):
    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError("Enter at least two coefficients.")

    values = []
    for part in parts:
        try:
            value = sp.sympify(part, rational=True)
        except (sp.SympifyError, TypeError, SyntaxError):
            raise ValueError(f"Invalid coefficient: {part}")

        if value.free_symbols:
            raise ValueError(
                "Coefficients must be numbers, not expressions with variables."
            )
        if value.is_real is not True or value.is_finite is not True:
            raise ValueError(
                "Enter finite real coefficients only. Complex coefficients are not allowed."
            )
        values.append(value)

    while len(values) > 1 and values[0] == 0:
        values.pop(0)

    if all(value == 0 for value in values):
        raise ValueError("The zero polynomial is not supported.")
    if len(values) < 2:
        raise ValueError("Enter a non-constant polynomial.")

    degree = len(values) - 1
    if degree > MAX_DEGREE:
        raise ValueError(
            f"Maximum supported degree is {MAX_DEGREE}. You entered degree {degree}."
        )
    return values


def readable_root(root_value):
    if type(root_value).__name__ == "ComplexRootOf":
        return sp.N(root_value, 10)
    return sp.simplify(root_value)


def all_roots_including_complex(polynomial):
    if polynomial.degree() <= 0:
        return []
    try:
        roots = list(polynomial.all_roots())
    except Exception:
        try:
            roots = list(polynomial.nroots(n=12, maxsteps=500))
        except Exception:
            coefficients = [complex(sp.N(c, 20)) for c in polynomial.all_coeffs()]
            numeric_roots = np.roots(coefficients)
            roots = [
                sp.Float(root.real, 10) + sp.Float(root.imag, 10) * sp.I
                for root in numeric_roots
            ]
    return [readable_root(root_value) for root_value in roots]


def real_roots_only(polynomial):
    if polynomial.is_zero or polynomial.degree() <= 0:
        return []
    try:
        return list(polynomial.real_roots())
    except Exception:
        roots = all_roots_including_complex(polynomial)
        real_roots = []
        for root_value in roots:
            if root_value.is_real is True:
                real_roots.append(sp.simplify(root_value))
            else:
                try:
                    number = complex(sp.N(root_value, 25))
                    tolerance = 1e-8 * (1 + abs(number.real))
                    if abs(number.imag) <= tolerance:
                        real_roots.append(sp.Float(number.real, 10))
                except Exception:
                    pass
        return real_roots


def equivalent(a, b):
    try:
        difference = sp.simplify(a - b)
        if difference == 0:
            return True
        if difference.is_number and not difference.free_symbols:
            return abs(complex(sp.N(difference, 25))) < 1e-7
        return False
    except Exception:
        return False


def parse_expression(text):
    text = text.strip().replace("^", "**")
    return sp.sympify(text, locals={"x": x, "I": sp.I}, rational=True)


def add_lesson(lessons, title, explanation, working, question, hint, answer, kind):
    lessons.append({
        "title": title, "explanation": explanation, "working": working,
        "question": question, "hint": hint, "answer": answer, "kind": kind
    })


def build_lesson(coefficients):
    lessons = []
    poly = sp.Poly.from_list(coefficients, gens=x)
    expr = poly.as_expr()
    degree = poly.degree()
    factorized = sp.factor(expr)
    roots = all_roots_including_complex(poly)
    first = sp.diff(expr, x)
    second = sp.diff(expr, x, 2)

    add_lesson(
        lessons, "Lesson 1 — Identify the degree",
        "The degree of a nonzero polynomial is the largest exponent of x with a nonzero coefficient.",
        f"f(x) = {sp.sstr(expr)}\nDegree = {degree}",
        "What is the degree of this polynomial?", "Find the highest power of x.",
        degree, "number"
    )

    constant = expr.subs(x, 0)
    add_lesson(
        lessons, "Lesson 2 — Find the constant term",
        "The constant term does not contain x. It is also f(0), the y-intercept.",
        f"f(0) = {sp.sstr(constant)}",
        "What is the constant term?",
        "Substitute x = 0. All terms containing x disappear.",
        constant, "expression"
    )

    add_lesson(
        lessons, "Lesson 3 — Form the equation",
        "To find the roots, set the polynomial equal to zero. A root is an x-value that makes the expression zero.",
        f"{sp.sstr(expr)} = 0",
        "What number should the polynomial equal when finding roots?",
        "We are looking for values that make the polynomial equal zero.",
        0, "number"
    )

    add_lesson(
        lessons, "Lesson 4 — Factorize",
        "Factorization expresses a polynomial as a product of simpler expressions.",
        f"{sp.sstr(expr)} = {sp.sstr(factorized)}",
        "Enter the factorized expression.",
        "Use SymPy-style notation. Example: (x-4)*(x+1). Multiplication signs are recommended.",
        factorized, "expression"
    )

    roots_text = ", ".join(sp.sstr(root_value) for root_value in roots)
    add_lesson(
        lessons, "Lesson 5 — Find all roots",
        "Use the zero-product property. Roots may be real or complex, but coefficients entered must be real.",
        f"All roots, including multiplicities:\n{roots_text}",
        "Enter all roots separated by commas. For x^2 + 1, enter I, -I.",
        "Solve each factor equal to zero. I is the imaginary unit, I**2 = -1. Include repeated roots.",
        roots, "roots"
    )

    add_lesson(
        lessons, "Lesson 6 — Differentiate",
        "The first derivative describes the rate at which the function changes. Differentiate each term using the power rule.",
        f"f'(x) = {sp.sstr(first)}",
        "What is the first derivative? Use x**2 for x squared.",
        "For x**n, multiply by n and reduce the power by one.",
        first, "expression"
    )

    critical_poly = sp.Poly(first, x)
    real_critical = real_roots_only(critical_poly)
    distinct_critical = []
    for value in real_critical:
        if not any(equivalent(value, old_value) for old_value in distinct_critical):
            distinct_critical.append(value)

    point_lines = []
    for c in distinct_critical:
        y_value = sp.simplify(expr.subs(x, c))
        second_value = sp.simplify(second.subs(x, c))
        if second_value.is_positive:
            classification = "local minimum"
        elif second_value.is_negative:
            classification = "local maximum"
        else:
            classification = "test inconclusive"
        point_lines.append(f"({sp.sstr(c)}, {sp.sstr(y_value)}) — {classification}")
    if not point_lines:
        point_lines = ["No real stationary points found."]

    add_lesson(
        lessons, "Lesson 7 — Investigate stationary points",
        "Stationary points are candidates obtained by solving f'(x) = 0. The second derivative can classify ordinary local maxima and minima, but sometimes the test is inconclusive.",
        "\n".join(point_lines),
        "How many distinct real stationary points does this polynomial have?",
        "Solve f'(x) = 0 and count its distinct real solutions.",
        len(distinct_critical), "number"
    )

    add_lesson(
        lessons, "Lesson 8 — Examine concavity",
        "The second derivative describes how the gradient changes. An inflection point requires a change in concavity.",
        f"f''(x) = {sp.sstr(second)}",
        "Enter the second derivative.",
        "Differentiate the first derivative once more.",
        second, "expression"
    )

    add_lesson(
        lessons, "Lesson 9 — Review your understanding",
        "You have examined the structure, roots and derivatives of the polynomial. These tools connect algebra to the shape of its graph.",
        f"Polynomial: {sp.sstr(expr)}\nDegree: {degree}\nFactorization: {sp.sstr(factorized)}\nAll roots: {roots_text}",
        "What is the degree of the polynomial?", "Return to the highest power of x.",
        degree, "number"
    )
    return lessons, poly, expr


def reset_course():
    for key in ("lessons", "poly", "expr", "current", "score", "attempted", "feedback", "revealed"):
        st.session_state.pop(key, None)


# Input panel
st.subheader("Polynomial Input")
with st.form("polynomial_form"):
    coefficient_text = st.text_input(
        "Enter coefficients from highest power to constant term",
        value=st.session_state.get("coefficient_text", "1 -3 -4"),
        help="Separate coefficients with spaces or commas. Fractions are supported."
    )
    st.caption("Real coefficients only • Roots may be complex • Maximum degree 20")
    submitted = st.form_submit_button("Start / Restart Lesson", type="primary")

if submitted:
    try:
        coefficients = parse_coefficients(coefficient_text)
        lessons, poly, expr = build_lesson(coefficients)
        st.session_state["coefficient_text"] = coefficient_text
        st.session_state["lessons"] = lessons
        st.session_state["poly"] = poly
        st.session_state["expr"] = expr
        st.session_state["current"] = 0
        st.session_state["score"] = 0
        st.session_state["attempted"] = []
        st.session_state["feedback"] = "Think about the question before answering."
        st.session_state["revealed"] = False
    except Exception as error:
        st.error(f"Input Error: {error}")

example_cols = st.columns(4)
examples = [
    ("Quadratic", "1 -3 -4"),
    ("Complex roots: x² + 1", "1 0 1"),
    ("20th degree: x²⁰ − 1", "1 " + "0 " * 19 + "-1"),
    ("Fractions", "1/2 -3/4 1/8"),
]
for col, (label, value) in zip(example_cols, examples):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["coefficient_text"] = value
            st.rerun()

if "lessons" not in st.session_state:
    st.info("Enter coefficients and click **Start / Restart Lesson**, or choose an example above.")
    st.stop()

lessons = st.session_state["lessons"]
poly = st.session_state["poly"]
expr = st.session_state["expr"]
current = st.session_state["current"]
item = lessons[current]

st.divider()
left, right = st.columns([3, 2], gap="large")

with left:
    st.progress((current + 1) / len(lessons), text=f"Question {current + 1} of {len(lessons)}")
    st.subheader(item["title"])
    st.markdown("**LEARN THE CONCEPT**")
    st.write(item["explanation"])
    st.markdown("**MATHEMATICAL WORKING**")
    st.code(item["working"], language="text")
    st.markdown("**YOUR QUESTION**")
    st.write(item["question"])

    answer = st.text_input(
        "Your answer",
        key=f"answer_{current}",
        placeholder="Type your answer here"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        check_clicked = st.button("Check Answer", type="primary", use_container_width=True)
    with c2:
        hint_clicked = st.button("Get Hint", use_container_width=True)
    with c3:
        reveal_clicked = st.button("Reveal Answer", use_container_width=True)

    if hint_clicked:
        st.info(item["hint"])

    if reveal_clicked:
        expected = item["answer"]
        if item["kind"] == "roots":
            answer_text = ", ".join(sp.sstr(value) for value in expected)
        else:
            answer_text = sp.sstr(expected)
        st.warning(f"Answer: {answer_text}")
        st.session_state["revealed"] = True

    if check_clicked:
        if not answer.strip():
            st.session_state["feedback"] = "Enter an answer first."
        else:
            try:
                kind = item["kind"]
                expected = item["answer"]
                if kind == "number":
                    user_value = parse_expression(answer)
                    if user_value.is_real is not True:
                        raise ValueError("This question expects a real numerical answer.")
                    correct = equivalent(user_value, expected)
                elif kind == "expression":
                    user_value = parse_expression(answer)
                    correct = equivalent(user_value, expected)
                elif kind == "roots":
                    user_roots = [
                        parse_expression(part.strip())
                        for part in answer.split(",") if part.strip()
                    ]
                    for root_value in user_roots:
                        if root_value.free_symbols or root_value.is_finite is not True:
                            raise ValueError("Enter numerical roots only.")
                    remaining = list(expected)
                    correct = len(user_roots) == len(remaining)
                    if correct:
                        for root_value in user_roots:
                            match = next(
                                (i for i, target in enumerate(remaining)
                                 if equivalent(root_value, target)),
                                None
                            )
                            if match is None:
                                correct = False
                                break
                            remaining.pop(match)
                else:
                    correct = False

                attempted = set(st.session_state["attempted"])
                if current not in attempted:
                    attempted.add(current)
                    st.session_state["attempted"] = sorted(attempted)
                    if correct:
                        st.session_state["score"] += 1

                st.session_state["feedback"] = (
                    "Correct! Well done. Read the explanation, then continue."
                    if correct else "Not quite. Review the hint and try again."
                )
            except Exception:
                st.session_state["feedback"] = "I could not understand that answer. Check your notation."

    feedback = st.session_state.get("feedback", "")
    if feedback:
        if feedback.startswith("Correct!"):
            st.success(feedback)
        elif feedback.startswith("Not quite") or feedback.startswith("I could not"):
            st.error(feedback)
        else:
            st.info(feedback)

    st.caption(
        f"Correct first attempts: {st.session_state['score']} / "
        f"{len(st.session_state['attempted'])} questions attempted"
    )

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("← Previous", disabled=current == 0, use_container_width=True):
            st.session_state["current"] -= 1
            st.session_state["feedback"] = "Think about the question before answering."
            st.rerun()
    with nav2:
        if st.button("Next Question →", disabled=current >= len(lessons) - 1, use_container_width=True):
            st.session_state["current"] += 1
            st.session_state["feedback"] = "Think about the question before answering."
            st.rerun()

with right:
    st.markdown("### VISUAL LEARNING")
    st.caption(
        "Red points indicate real roots. Complex roots appear in Lesson 5, "
        "but cannot be shown as x-intercepts on this real graph."
    )

    real_roots = real_roots_only(poly)
    numeric_real_roots = []
    for root_value in real_roots:
        try:
            numeric_real_roots.append(float(sp.N(root_value)))
        except Exception:
            pass

    if numeric_real_roots:
        left_edge = min(numeric_real_roots) - 3
        right_edge = max(numeric_real_roots) + 3
    else:
        left_edge, right_edge = -10, 10

    xs = np.linspace(left_edge, right_edge, 1200)
    function = sp.lambdify(x, expr, "numpy")
    with np.errstate(all="ignore"):
        ys = np.asarray(function(xs), dtype=float)
    ys[~np.isfinite(ys)] = np.nan
    ys[np.abs(ys) > 1e6] = np.nan

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(xs, ys, color=BLUE, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    for root_value in numeric_real_roots:
        ax.scatter(root_value, 0, color="red", zorder=5)
    ax.set_title(f"f(x) = {sp.sstr(expr)}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Polynomial Summary")
    st.write(f"**Polynomial:** `{sp.sstr(expr)}`")
    st.write(f"**Degree:** {poly.degree()}")
    st.write(f"**Factorization:** `{sp.sstr(sp.factor(expr))}`")
    roots = all_roots_including_complex(poly)
    st.write("**All roots (including multiplicities):**")
    st.code(", ".join(sp.sstr(r) for r in roots), language="text")
