import React, { useState } from "react";
import "./App.css";

function App() {

  const [vehicles,setVehicles] = useState([]);
  const [selected,setSelected] = useState(null);

  const addVehicle = () => {

    if(vehicles.length >= 20){
      alert("Maximum 20 vehicles allowed");
      return;
    }

    const types = ["Car","Bus","Ambulance","Truck","Bike"];

    const newVehicle = {
      id: Date.now(),
      name: "Vehicle " + (vehicles.length+1),
      type: types[Math.floor(Math.random()*types.length)],
      speed: Math.floor(Math.random()*120),
      weather: "Clear",
      posx: Math.floor(Math.random()*100),
      posy: Math.floor(Math.random()*100),
      attack: Math.random()>0.8 ? "Unsafe":"Safe"
    };

    setVehicles([...vehicles,newVehicle]);
  };

  const removeVehicle = (id)=>{
    setVehicles(vehicles.filter(v=>v.id!==id));
    if(selected?.id===id){
      setSelected(null);
    }
  };

  return (

    <div className="container">

      <h1>🚗 Vehicle Network Dashboard</h1>

      <button onClick={addVehicle}>Add Vehicle</button>

      <table>

        <thead>

          <tr>
            <th>Vehicle</th>
            <th>Type</th>
            <th>Speed</th>
            <th>Pos X</th>
            <th>Pos Y</th>
            <th>Status</th>
            <th>Action</th>
          </tr>

        </thead>

        <tbody>

        {vehicles.map(v=>(
          <tr
            key={v.id}
            className={v.attack==="Unsafe" ? "danger":""}
          >

          <td onClick={()=>setSelected(v)}>{v.name}</td>
          <td>{v.type}</td>
          <td>{v.speed}</td>
          <td>{v.posx}</td>
          <td>{v.posy}</td>
          <td>{v.attack}</td>

          <td>
            <button onClick={()=>removeVehicle(v.id)}>
              Remove
            </button>
          </td>

          </tr>
        ))}

        </tbody>

      </table>

      {selected && (

        <div className="dashboard">

          <h2>Vehicle Dashboard</h2>

          <p><b>Name:</b> {selected.name}</p>
          <p><b>Type:</b> {selected.type}</p>
          <p><b>Speed:</b> {selected.speed}</p>
          <p><b>Position:</b> {selected.posx},{selected.posy}</p>
          <p><b>Status:</b> {selected.attack}</p>

        </div>

      )}

    </div>

  );
}

export default App;