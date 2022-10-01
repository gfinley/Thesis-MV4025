
var Mode = {
    SelectUnit : 0,
    TakeAction : 1
};

var mode = Mode.SelectUnit;
var selectedUnit = null;

var markIndex = {};

function resetGuiState() {
    mode = Mode.SelectUnit;
    SVGGui.clearMarks();
}

function hexMouseDownHandler(evt) {
    if (mode === Mode.TakeAction) {
        let hex = Map.hexIndex[this.id];
        if (!_setup_phase) {
            if (!moveTargets.includes(hex))
                return;
            moveTargets = [];
            fireTargets = [];
            SVGGui.clearMarks();
            selectedUnit.setHex(hex);
            SVGUnitSymbol.moveSymbolToHex(selectedUnit.uniqueId, hex);
            mode = Mode.SelectUnit;
        }
        else { // _setup_phase
            // If a hex that is tagged as startup for same faction, move
            if (hex.setup && hex.setup.substr(11) === _faction) {
               sendSetupMove(hex);
               mode = Mode.SelectUnit;
            }
        }
    }
}

function hexMouseOverHandler(evt) {
}

function edgeMouseOver(evt) {
}

function markerMouseDown(evt) {
    if (_on_move) {
        let markData = markIndex[this.id];
        if (markData.type==="hex") {
            console.log("Send move request to server, target hex id: "+markData.value.id);
            sendMove(markData.value);
        }
        else if (markData.type==="unit") {
            console.log("Send shoot request to server, target id: "+markData.value.uniqueId);
            sendFire(markData.value);
        }
        else if (markData.type==="self") {
            SVGGui.clearMarks();
        }
        mode = Mode.SelectUnit;
    }
}

function unitMouseDownHandler(evt) {
    if (mode === Mode.SelectUnit) {
        selectedUnit = Unit.unitIndex[this.id];
        if (!selectedUnit.canMove)  return;
        let markID = "mark "+selectedUnit.hex.id;
        SVGGui.markHex(selectedUnit.hex,"green",markID);
        markIndex[markID] = {type:"self", value:selectedUnit.hex};
        mode = Mode.TakeAction;
        if (!_setup_phase) {
            for (let hex of selectedUnit.findMoveTargets()) {
                let markID = "mark "+hex.id;
                SVGGui.markHex(hex,"blue",markID);
                markIndex[markID] = {type:"hex", value:hex};
            }
            for (let unit of selectedUnit.findFireTargets()) {
                let markID = "mark "+unit.uniqueId;
                SVGGui.markHex(unit.hex,"red",markID);
                markIndex[markID] = {type:"unit", value:unit};
            }
        }
    }
    else if (mode === Mode.TakeAction) {
        let exchangeTarget = Unit.unitIndex[this.id];
        if (_faction == exchangeTarget.faction) {
            sendSetupExchange(exchangeTarget);
            mode = Mode.SelectUnit;
        }
    }
}
