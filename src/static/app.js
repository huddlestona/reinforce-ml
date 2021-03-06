let get_submission = function() {
    let u = $("textarea#user-submission").val();
    let word = $("span#word").text();
    return {'user_subm': u, 'word': word}
};

let send_submission_json = function(submission) {
    $.ajax({
        url: '/submit',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        data: JSON.stringify(submission),
        success: function (data) {
            display_results(data);
        }
    });
};

let display_results = function(results) {
    let user_submission = $("div#submission")
    user_submission.html(results.submission)
    let answer = $("div#answer")
    answer.html(results.answer)
    let score = $("div#score")
    score.html(results.score)
    let leaderboard = $("div#leaderboard")
    leaderboard.html(results.leaderboard)
    let next_step = $("div#next_step")
    next_step.html(results.next_step)
};

$(document).ready(function() {
    $("button#submit").click(function() {
        let submission = get_submission();
        send_submission_json(submission);
    })
})
