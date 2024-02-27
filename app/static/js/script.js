var board = null
var game = new Chess()
var $status = $('#status')
var $fen = $('#fen')
var $pgn = $('#pgn')
var $msg = $('#message')
var $comment = $('#comment')

function onDragStart (source, piece, position, orientation) {
  // do not pick up pieces if the game is over
  if (game.isGameOver()) return false

  // only pick up pieces for the side to move
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false
  }
}

function displayMessage(message) {
    $status.html(message); // Update the content of the message div
}

function onDrop (source, target) {
    try {
        // see if the move is legal
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' // NOTE: always promote to a queen for example simplicity
        });

        // illegal move
        if (move === null) {
            throw new Error('Invalid move');
        }

        updateStatus();

        updateFen(game.fen());

    } catch (error) {
        // Display the error message to the user
        displayMessage("This move is not allowed. Please try again.");
        // Log the error to the console for debugging
        console.error(error);
    }
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd () {
  board.position(game.fen())
}

function updateStatus () {
  var status = ''

  var moveColor = 'White'
  if (game.turn() === 'b') {
    moveColor = 'Black'
  }

  // checkmate?
  if (game.isCheckmate()) {
    status = 'Game over, ' + moveColor + ' is in checkmate.'
  }

  // draw?
  else if (game.isDraw()) {
    status = 'Game over, drawn position'
  }

  // game still on
  else {
    status = moveColor + ' to move'

    // check?
    if (game.isCheck()) {
      status += ', ' + moveColor + ' is in check'
    }
  }

  $status.html(status)
  $fen.html(game.fen())
  $pgn.html(game.pgn());
}

var config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
}
board = Chessboard('myBoard', config)

updateStatus()

// Function to send the updated FEN data to the backend and receive HTML rendering
async function updateFen(fen) {
  try {
    const response = await fetch('/submit_fen', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `fen=${encodeURIComponent(fen)}`,
    });

    if (response.ok) {
      const responseData = await response.text();
      console.log(responseData)
      document.getElementById("comment").innerHTML = responseData; // Render the HTML received from the backend
    } else {
      console.error('Failed to update FEN:', response.statusText);
    }
  } catch (error) {
    console.error('Error updating FEN:', error);
  }
}

