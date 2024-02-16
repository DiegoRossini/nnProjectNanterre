
// Initialize chess board
var board = Chessboard("board", "start");

// Create a new chess game instance
var game = new Chess();

// Function to handle user moves
function handleMove(source, target) {
    // Validate the move using chess.js
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q' // promote to queen for simplicity
    });

    // If the move is legal, update the board
    if (move !== null) {
        board.position(game.fen());
    } else {
        // If the move is illegal, display an error message
        alert("Invalid move. Please make a legal move.");
        return 'snapback'; // Snap the piece back to its original position
    }
}

// Set up event handlers to detect user moves
board.on('dragStart', function (source, piece, position, orientation) {
    // If it's not the user's turn or the game is over, prevent dragging
    if (game.game_over() === true || (game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
});

board.on('drop', function (source, target) {
    // If it's not the user's turn or the game is over, prevent dropping
    if (game.game_over() === true || (game.turn() === 'w' && source.search(/^b/) !== -1) ||
        (game.turn() === 'b' && source.search(/^w/) !== -1)) {
        return 'snapback';
    }

    // Handle the move
    handleMove(source, target);
});